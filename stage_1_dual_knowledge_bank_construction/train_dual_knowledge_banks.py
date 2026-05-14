#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from eviscreen.knowledge_bank import backbones, common, sampler, utils
from eviscreen.knowledge_bank.datasets.fundus import DatasetSplit, FundusDataset
from eviscreen.knowledge_bank.knowledge_bank import Knowledge_Bank
from eviscreen.knowledge_bank.pos_embed import interpolate_pos_embed


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EviScreen fundus dual knowledge banks.")
    parser.add_argument("--data-root", required=True, type=Path, help="Dataset root containing fundus CSV folders.")
    parser.add_argument("--raw-root", required=True, type=Path, help="Raw dataset root containing preprocessed image folders.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="RETFound_dinov2_meh.pth checkpoint.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to save normal and pathological knowledge banks.")
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--chunk-size", default=5000, type=int)
    parser.add_argument("--train-scale", default=5000, type=int)
    parser.add_argument("--resize", default=224, type=int)
    parser.add_argument("--imagesize", default=224, type=int)
    parser.add_argument("--coreset-percentage", default=0.1, type=float)
    parser.add_argument("--debug-samples", default=None, type=int, help="Randomly sample this many training images per knowledge bank.")
    parser.add_argument("--faiss-on-gpu", action="store_true")
    return parser.parse_args()


def load_ret_found_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "teacher" in checkpoint:
        checkpoint = checkpoint["teacher"]
    renamed = {}
    for key, value in checkpoint.items():
        renamed[key.replace("backbone.", "") if key.startswith("backbone.") else key] = value
    interpolate_pos_embed(model, renamed)
    msg = model.load_state_dict(renamed, strict=False)
    LOGGER.info("Loaded checkpoint with message: %s", msg)


def build_knowledge_bank(args: argparse.Namespace, category: str, save_dir: Path, device: torch.device) -> None:
    dataset = FundusDataset(
        data_root=args.data_root,
        raw_root=args.raw_root,
        classname="fundus",
        resize=args.resize,
        imagesize=args.imagesize,
        split=DatasetSplit.TRAIN,
        train_scale=args.train_scale,
        category=category,
    )
    if args.debug_samples is not None:
        rng = np.random.default_rng(args.seed)
        sample_count = min(args.debug_samples, len(dataset))
        indices = rng.choice(len(dataset), size=sample_count, replace=False)
        dataset = Subset(dataset, indices.tolist())
        LOGGER.info("Debug mode: sampled %s images for %s knowledge bank.", sample_count, category)

    input_shape = dataset.dataset.imagesize if isinstance(dataset, Subset) else dataset.imagesize
    dataloader_params = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    backbone = backbones.load("vit_large_patch14_dinov2meh")
    load_ret_found_checkpoint(backbone, args.checkpoint)
    backbone.name = "vit_large_patch14_dinov2meh"
    backbone.seed = None

    feature_sampler = sampler.ApproximateGreedyCoresetSampler(args.coreset_percentage, device)
    nn_method = common.FaissNN(args.faiss_on_gpu, args.num_workers)
    bank = Knowledge_Bank(device)
    bank.load(
        backbone=backbone,
        layers_to_extract_from=["blocks.7", "blocks.17"],
        device=device,
        input_shape=input_shape,
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        featuresampler=feature_sampler,
        anomaly_scorer_num_nn=1,
        nn_method=nn_method,
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    all_sampled_features = []
    bank.forward_modules.eval()
    num_chunks = (len(dataset) + args.chunk_size - 1) // args.chunk_size
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * args.chunk_size
        end_idx = min(start_idx + args.chunk_size, len(dataset))
        subset = Subset(dataset, range(start_idx, end_idx))
        loader = DataLoader(subset, **dataloader_params)

        chunk_features_list = []
        chunk_metadata_list = []
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"{category} chunk {chunk_idx + 1}/{num_chunks}")):
            images = batch["image"].to(torch.float).to(device)
            with torch.no_grad():
                features, metadata = bank._embed(images, provide_patch_metadata=True)
            global_image_indices = list(range(start_idx + batch_idx * args.batch_size, start_idx + batch_idx * args.batch_size + len(batch["image"])))
            for meta in metadata:
                meta["global_image_idx"] = global_image_indices[meta["image_idx"]]
                meta["chunk_idx"] = chunk_idx
            chunk_features_list.append(features)
            chunk_metadata_list.extend(metadata)

        chunk_features = np.concatenate(chunk_features_list, axis=0)
        sampled_features, sampled_indices = feature_sampler.run_with_indices(chunk_features)
        all_sampled_features.append(sampled_features)
        np.save(save_dir / f"sampled_chunk_features_{chunk_idx}.npy", sampled_features)
        np.save(save_dir / f"sampled_chunk_metadata_{chunk_idx}.npy", [chunk_metadata_list[i] for i in sampled_indices])
        del chunk_features, chunk_features_list, sampled_features, sampled_indices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_memory_bank = np.concatenate(all_sampled_features, axis=0)
    bank.anomaly_scorer.fit(detection_features=[final_memory_bank])
    bank.save_to_path(str(save_dir))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    utils.fix_seeds(args.seed)
    device = utils.set_torch_device([args.gpu] if torch.cuda.is_available() else [])
    build_knowledge_bank(args, "normal", args.output_dir / "normal", device)
    build_knowledge_bank(args, "abnormal", args.output_dir / "pathological", device)


if __name__ == "__main__":
    main()
