#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from timm.optim import optim_factory
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
EVISCREEN_ROOT = SCRIPT_DIR.parent
STAGE1_ROOT = EVISCREEN_ROOT / "stage_1_dual_knowledge_bank_construction"
REPRODUCE_DIR = EVISCREEN_ROOT / "reproduce_directly"
sys.path.insert(0, str(STAGE1_ROOT))
sys.path.insert(0, str(REPRODUCE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from eviscreen.knowledge_bank import common  # noqa: E402
from retrieved_distance_dataset import RetrievedDistanceDataset, load_patchcore_index_proxies  # noqa: E402
from retrieving_distance_cat_head import MainClassifier as RetrievingDistanceCatHead  # noqa: E402


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EviScreen stage-2 RetrievingDistanceCatHead.")
    parser.add_argument("--retrieved-data-root", required=True, type=Path)
    parser.add_argument("--normal-bank", required=True, type=Path)
    parser.add_argument("--pathological-bank", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)

    parser.add_argument("--model", default="RetrievingDistanceCatHead")
    parser.add_argument("--modality", default="fundus")
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--note", default="original_range_disdim1024")
    parser.add_argument("--script", default="")

    parser.add_argument("--num-patches", default=256, type=int)
    parser.add_argument("--embed-dim", default=1024, type=int)
    parser.add_argument("--num-heads", default=8, type=int)
    parser.add_argument("--depth", default=4, type=int)

    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("--warmup-epochs", default=10, type=int)
    parser.add_argument("--accum-iter", default=1, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num-workers", default=1, type=int)
    parser.add_argument("--save-freq", default=1, type=int)
    parser.add_argument("--blr", default=8e-4, type=float)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--min-lr", default=0.0, type=float)
    parser.add_argument("--weight-decay", default=0.05, type=float)
    parser.add_argument("--temperature", default=0.07, type=float)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--from-begin", action="store_true")
    parser.add_argument("--test-sets", nargs="+", default=["JSIEC_original", "RIADD_original"])
    parser.add_argument("--max-train-samples", default=None, type=int)
    parser.add_argument("--max-val-samples", default=None, type=int)
    parser.add_argument("--max-test-samples", default=None, type=int)
    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def require_bank(path: Path, description: str) -> None:
    require_file(path / "nnscorer_search_index.faiss", f"{description} FAISS index")


def require_retrieved_split(root: Path, dataset_name: str) -> None:
    for path in (
        root / f"{dataset_name}_normal",
        root / f"{dataset_name}_abnormal",
        root / f"{dataset_name}_distances.npy",
        root / f"{dataset_name}_abnormal_distances.npy",
        root / f"{dataset_name}_anomaly_labels.npy",
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing retrieved-data input for {dataset_name}: {path}")


def setup_output_dir(args: argparse.Namespace, base_batch_size: int) -> None:
    if args.output_dir:
        if args.note != "debug":
            args.output_dir = args.output_dir / (
                f"pretrain_MODEL{args.model}_BACKBONE{args.backbone}_TEMP{args.temperature}_"
                f"EP{args.epochs}_WM{args.warmup_epochs}_LR{args.lr}_BS{base_batch_size}_"
                f"MODALITY{args.modality}_{args.note}"
            )
        else:
            args.output_dir = args.output_dir / "debug"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.script:
            script_path = Path(args.script)
            if script_path.exists():
                shutil.copy(script_path, args.output_dir)


def adjust_learning_rate(optimizer, epoch_progress: float, args: argparse.Namespace) -> float:
    if epoch_progress < args.warmup_epochs:
        lr = args.lr * epoch_progress / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (epoch_progress - args.warmup_epochs) / (args.epochs - args.warmup_epochs))
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def move_batch_to_device(batch, device: torch.device):
    (
        cur_features,
        cur_retrieved_features_from_normal,
        cur_distances_from_normal,
        cur_retrieved_features_from_abnormal,
        cur_distances_from_abnormal,
        labels,
    ) = batch
    cur_retrieved_features_from_normal = cur_retrieved_features_from_normal.to(device, non_blocking=True)
    cur_retrieved_features_from_abnormal = cur_retrieved_features_from_abnormal.to(device, non_blocking=True)
    if device.type != "cuda":
        cur_retrieved_features_from_normal = cur_retrieved_features_from_normal.float()
        cur_retrieved_features_from_abnormal = cur_retrieved_features_from_abnormal.float()

    return (
        cur_features.to(device, non_blocking=True),
        cur_retrieved_features_from_normal,
        cur_distances_from_normal.to(device, non_blocking=True),
        cur_retrieved_features_from_abnormal,
        cur_distances_from_abnormal.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
    )


def forward_loss(model, batch, device: torch.device, criterion):
    (
        cur_features,
        cur_retrieved_features_from_normal,
        cur_distances_from_normal,
        cur_retrieved_features_from_abnormal,
        cur_distances_from_abnormal,
        labels,
    ) = move_batch_to_device(batch, device)
    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        logits = model(
            cur_features,
            cur_retrieved_features_from_normal,
            cur_distances_from_normal,
            cur_retrieved_features_from_abnormal,
            cur_distances_from_abnormal,
        ).squeeze(-1)
        loss = criterion(logits, labels.float())
    return logits, loss, labels


def train_one_epoch(model, dataloader, optimizer, device: torch.device, epoch: int, scaler, args: argparse.Namespace) -> dict:
    model.train(True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer.zero_grad()
    total_loss = 0.0
    steps = 0
    start = time.time()

    for data_iter_step, batch in enumerate(dataloader):
        if data_iter_step % args.accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(dataloader) + epoch, args)

        _, loss, _ = forward_loss(model, batch, device, criterion)
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Loss is {loss_value}, stopping training.")

        loss = loss / args.accum_iter
        scaler.scale(loss).backward()
        if (data_iter_step + 1) % args.accum_iter == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss_value
        steps += 1
        if data_iter_step % 10 == 0 or data_iter_step == len(dataloader) - 1:
            lr = optimizer.param_groups[0]["lr"]
            LOGGER.info("Epoch %s [%s/%s] loss=%.6f lr=%.8f", epoch, data_iter_step, len(dataloader), loss_value, lr)

    if device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "loss": total_loss / max(steps, 1),
        "lr": optimizer.param_groups[0]["lr"],
        "epoch_time_sec": time.time() - start,
    }


@torch.no_grad()
def validate(model, dataloader, device: torch.device, split_name: str) -> dict:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        logits, loss, labels = forward_loss(model, batch, device, criterion)
        preds = torch.sigmoid(logits)
        total_loss += float(loss.item())
        all_preds.append(preds.float().cpu())
        all_labels.append(labels.cpu())

    y_true = torch.cat(all_labels).numpy().astype(np.int64)
    y_score = torch.cat(all_preds).numpy().astype(float)
    stats = {
        "loss": total_loss / max(len(dataloader), 1),
        "val_auroc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.0,
        "val_auprc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.0,
        "num_images": int(len(y_true)),
    }
    LOGGER.info("%s stats: %s", split_name, stats)
    return stats


def save_checkpoint(args, model, optimizer, scaler, epoch: int, name: str) -> None:
    if not args.output_dir:
        return
    checkpoint_path = args.output_dir / f"checkpoint-{name}.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "name": name,
            "scaler": scaler.state_dict(),
            "args": args,
        },
        checkpoint_path,
    )
    LOGGER.info("Saved checkpoint: %s", checkpoint_path)


def load_checkpoint(args, model, optimizer, scaler) -> None:
    if not args.resume:
        return
    checkpoint = torch.load(args.resume, map_location="cpu")
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    LOGGER.info("Resume checkpoint %s: %s", args.resume, msg)
    if not args.from_begin and not args.eval and "optimizer" in checkpoint and "epoch" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        args.start_epoch = int(checkpoint["epoch"]) + 1
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])


def build_loader(dataset, args: argparse.Namespace, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if args.model != "RetrievingDistanceCatHead":
        raise ValueError(f"Only RetrievingDistanceCatHead is supported here, got {args.model}.")
    if args.modality != "fundus":
        raise ValueError(f"Only fundus modality is supported here, got {args.modality}.")

    require_bank(args.normal_bank, "Normal knowledge bank")
    require_bank(args.pathological_bank, "Pathological knowledge bank")
    require_retrieved_split(args.retrieved_data_root, "fundus_remain_5000")
    require_retrieved_split(args.retrieved_data_root, "fundus_val")

    base_batch_size = args.batch_size
    eff_batch_size = args.batch_size * args.accum_iter
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    setup_output_dir(args, base_batch_size)

    LOGGER.info("base lr: %.2e", args.lr * 256 / eff_batch_size)
    LOGGER.info("actual lr: %.2e", args.lr)
    LOGGER.info("output dir: %s", args.output_dir)
    LOGGER.info("%s", args)

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    patchcore_instance_normal, patchcore_instance_abnormal = load_patchcore_index_proxies(
        common, args.normal_bank, args.pathological_bank
    )

    train_dataset = None
    val_dataset = None
    if not args.eval:
        train_dataset = RetrievedDistanceDataset(
            split="train",
            retrieved_data_root=args.retrieved_data_root,
            patchcore_instance_normal=patchcore_instance_normal,
            patchcore_instance_abnormal=patchcore_instance_abnormal,
            limit=args.max_train_samples,
        )
        val_dataset = RetrievedDistanceDataset(
            split="val",
            retrieved_data_root=args.retrieved_data_root,
            patchcore_instance_normal=patchcore_instance_normal,
            patchcore_instance_abnormal=patchcore_instance_abnormal,
            limit=args.max_val_samples,
        )
        train_loader = build_loader(train_dataset, args, shuffle=True)
        val_loader = build_loader(val_dataset, args, shuffle=False)
    else:
        train_loader = None
        val_loader = None

    test_loaders = {}
    for split_name in args.test_sets:
        try:
            dataset = RetrievedDistanceDataset(
                split=split_name,
                retrieved_data_root=args.retrieved_data_root,
                patchcore_instance_normal=patchcore_instance_normal,
                patchcore_instance_abnormal=patchcore_instance_abnormal,
                limit=args.max_test_samples,
            )
        except FileNotFoundError as error:
            LOGGER.warning("Skipping %s because retrieved data is missing: %s", split_name, error)
            continue
        test_loaders[split_name] = build_loader(dataset, args, shuffle=False)

    model = RetrievingDistanceCatHead(
        embed_dim=args.embed_dim,
        num_patches=args.num_patches,
        num_heads=args.num_heads,
        depth=args.depth,
        args=args,
    ).to(device)
    optimizer = torch.optim.AdamW(
        optim_factory.param_groups_weight_decay(model, weight_decay=args.weight_decay, no_weight_decay_list=model.no_weight_decay()),
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    load_checkpoint(args, model, optimizer, scaler)

    if args.eval:
        for split_name, loader in test_loaders.items():
            validate(model, loader, device, split_name)
        return

    best_val_auroc = 0.0
    train_start = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        val_stats = validate(model, val_loader, device, "fundus_val")
        if val_stats["val_auroc"] > best_val_auroc:
            best_val_auroc = val_stats["val_auroc"]
            save_checkpoint(args, model, optimizer, scaler, epoch, "best_auroc")

        train_stats = train_one_epoch(model, train_loader, optimizer, device, epoch, scaler, args)
        save_checkpoint(args, model, optimizer, scaler, epoch, "newest")
        if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs:
            save_checkpoint(args, model, optimizer, scaler, epoch, "interval_save")

        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}
        if args.output_dir:
            with (args.output_dir / "log.txt").open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - train_start
    LOGGER.info("Training time %s", str(datetime.timedelta(seconds=int(total_time))))


if __name__ == "__main__":
    main()
