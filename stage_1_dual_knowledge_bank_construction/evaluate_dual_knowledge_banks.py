#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from eviscreen.knowledge_bank import common, utils
from eviscreen.knowledge_bank.datasets.fundus import DatasetSplit, FundusDataset
from eviscreen.knowledge_bank.knowledge_bank import Knowledge_Bank


LOGGER = logging.getLogger(__name__)
SPLITS = {
    "JSIEC_original": DatasetSplit.JSIEC_ORIGINAL,
    "RIADD_original": DatasetSplit.RIADD_ORIGINAL,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EviScreen dual knowledge banks.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--normal-bank", required=True, type=Path)
    parser.add_argument("--pathological-bank", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--test-sets", nargs="+", default=["JSIEC_original", "RIADD_original"], choices=sorted(SPLITS))
    parser.add_argument("--nn", default=16, type=int, help="Nearest neighbours used when loading each knowledge bank.")
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--resize", default=224, type=int)
    parser.add_argument("--imagesize", default=224, type=int)
    parser.add_argument("--max-test-samples", default=None, type=int, help="Evaluate only the first N images per test set.")
    parser.add_argument("--faiss-on-gpu", action="store_true")
    return parser.parse_args()


def load_bank(path: Path, checkpoint: Path, nn: int, device: torch.device, faiss_on_gpu: bool, num_workers: int) -> Knowledge_Bank:
    nn_method = common.FaissNN(faiss_on_gpu, num_workers)
    bank = Knowledge_Bank(device)
    bank.load_from_path(
        load_path=str(path),
        device=device,
        nn_method=nn_method,
        anomaly_scorer_num_nn=nn,
        backbone_path=str(checkpoint),
    )
    return bank


def evaluate_split(args: argparse.Namespace, split_name: str, normal_bank: Knowledge_Bank, pathological_bank: Knowledge_Bank, device: torch.device) -> dict:
    dataset = FundusDataset(
        data_root=args.data_root,
        raw_root=args.raw_root,
        classname="fundus",
        resize=args.resize,
        imagesize=args.imagesize,
        split=SPLITS[split_name],
    )
    img_paths = dataset.img_paths
    targets = dataset.targets
    if args.max_test_samples is not None:
        sample_count = min(args.max_test_samples, len(dataset))
        img_paths = dataset.img_paths[:sample_count]
        targets = dataset.targets[:sample_count]
        dataset = Subset(dataset, list(range(sample_count)))
        LOGGER.info("Debug mode: evaluating first %s images from %s.", sample_count, split_name)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    normal_scores, normal_patch_scores, *_ = normal_bank.predict(loader)
    pathological_scores, pathological_patch_scores, *_ = pathological_bank.predict(loader)
    y_true = np.asarray(targets, dtype=int)
    normal_scores = np.asarray(normal_scores, dtype=float)
    pathological_scores = np.asarray(pathological_scores, dtype=float)

    normal_patch_scores = np.asarray(normal_patch_scores, dtype=float)
    pathological_patch_scores = np.asarray(pathological_patch_scores, dtype=float)
    delta = normal_patch_scores - pathological_patch_scores
    delta[delta < 0] = 0
    delta_scores = delta.reshape(delta.shape[0], -1).mean(axis=1)

    metrics = {
        "normal_auroc": float(roc_auc_score(y_true, normal_scores)) if len(np.unique(y_true)) > 1 else None,
        "pathological_auroc": float(roc_auc_score(y_true, pathological_scores)) if len(np.unique(y_true)) > 1 else None,
        "delta_auroc": float(roc_auc_score(y_true, delta_scores)) if len(np.unique(y_true)) > 1 else None,
        "num_images": int(len(y_true)),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"{split_name}_predictions.csv"
    with result_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "y_true", "normal_score", "pathological_score", "delta_score"])
        for image_path, label, normal_score, pathological_score, delta_score in zip(img_paths, y_true, normal_scores, pathological_scores, delta_scores):
            writer.writerow([image_path, int(label), float(normal_score), float(pathological_score), float(delta_score)])

    metrics_path = args.output_dir / f"{split_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    LOGGER.info("%s metrics: %s", split_name, metrics)
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    utils.fix_seeds(args.seed)
    device = utils.set_torch_device([args.gpu] if torch.cuda.is_available() else [])
    normal_bank = load_bank(args.normal_bank, args.checkpoint, args.nn, device, args.faiss_on_gpu, args.num_workers)
    pathological_bank = load_bank(args.pathological_bank, args.checkpoint, args.nn, device, args.faiss_on_gpu, args.num_workers)
    summary = {split: evaluate_split(args, split, normal_bank, pathological_bank, device) for split in args.test_sets}
    (args.output_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
