#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


SCRIPT_DIR = Path(__file__).resolve().parent
EVISCREEN_ROOT = SCRIPT_DIR.parent
STAGE1_ROOT = EVISCREEN_ROOT / "stage_1_dual_knowledge_bank_construction"
sys.path.insert(0, str(STAGE1_ROOT))

from eviscreen.knowledge_bank import common, utils  # noqa: E402
from eviscreen.knowledge_bank.datasets.fundus import DatasetSplit, FundusDataset  # noqa: E402
from eviscreen.knowledge_bank.knowledge_bank import Knowledge_Bank  # noqa: E402


LOGGER = logging.getLogger(__name__)

SPLITS = {
    "JSIEC_original": DatasetSplit.JSIEC_ORIGINAL,
    "RIADD_original": DatasetSplit.RIADD_ORIGINAL,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract retrieved evidence for direct EviScreen stage-2 inference.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--normal-bank", required=True, type=Path)
    parser.add_argument("--pathological-bank", required=True, type=Path)
    parser.add_argument("--backbone-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory where retrieved_data files are written.")
    parser.add_argument("--test-sets", nargs="+", default=["JSIEC_original", "RIADD_original"], choices=sorted(SPLITS))
    parser.add_argument("--nn", default=16, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--resize", default=224, type=int)
    parser.add_argument("--imagesize", default=224, type=int)
    parser.add_argument("--max-test-samples", default=None, type=int)
    parser.add_argument("--faiss-on-gpu", action="store_true")
    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def require_bank(path: Path, description: str) -> None:
    require_file(path / "knowledge_bank_params.pkl", f"{description} params")
    require_file(path / "nnscorer_search_index.faiss", f"{description} FAISS index")


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


def to_sample_array(values, dtype=np.float32) -> np.ndarray:
    return np.asarray(values, dtype=dtype)


def extract_split(
    args: argparse.Namespace,
    split_name: str,
    normal_bank: Knowledge_Bank,
    pathological_bank: Knowledge_Bank,
) -> None:
    dataset = FundusDataset(
        data_root=args.data_root,
        raw_root=args.raw_root,
        classname="fundus",
        resize=args.resize,
        imagesize=args.imagesize,
        split=SPLITS[split_name],
    )
    targets = np.asarray(dataset.targets, dtype=np.int64)
    if args.max_test_samples is not None:
        sample_count = min(args.max_test_samples, len(dataset))
        dataset = Subset(dataset, list(range(sample_count)))
        targets = targets[:sample_count]
        LOGGER.info("Debug mode: extracting first %s images from %s.", sample_count, split_name)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    split_root = args.output_dir
    normal_save_dir = split_root / f"{split_name}_normal"
    pathological_save_dir = split_root / f"{split_name}_abnormal"

    LOGGER.info("Extracting %s evidence from normal bank into %s.", split_name, normal_save_dir)
    normal_scores, normal_patch_scores, *_normal_unused, normal_distances = normal_bank.predict(loader, save_path=str(normal_save_dir))

    LOGGER.info("Extracting %s evidence from pathological bank into %s.", split_name, pathological_save_dir)
    pathological_scores, pathological_patch_scores, *_pathological_unused, pathological_distances = pathological_bank.predict(
        loader,
        save_path=str(pathological_save_dir),
    )

    normal_patch_scores = to_sample_array(normal_patch_scores)
    pathological_patch_scores = to_sample_array(pathological_patch_scores)
    normal_distances = to_sample_array(normal_distances)
    pathological_distances = to_sample_array(pathological_distances)

    expected_shape = (len(targets), args.num_patches if hasattr(args, "num_patches") else 256, args.nn)
    if normal_distances.shape != expected_shape:
        raise RuntimeError(f"Unexpected normal distance shape for {split_name}: {normal_distances.shape}, expected {expected_shape}.")
    if pathological_distances.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected pathological distance shape for {split_name}: {pathological_distances.shape}, expected {expected_shape}."
        )

    split_root.mkdir(parents=True, exist_ok=True)
    np.save(split_root / f"{split_name}_patch_scores.npy", normal_patch_scores)
    np.save(split_root / f"{split_name}_distances.npy", normal_distances)
    np.save(split_root / f"{split_name}_abnormal_patch_scores.npy", pathological_patch_scores)
    np.save(split_root / f"{split_name}_abnormal_distances.npy", pathological_distances)
    np.save(split_root / f"{split_name}_anomaly_labels.npy", targets)

    delta = normal_patch_scores - pathological_patch_scores
    delta[delta < 0] = 0
    delta_scores = delta.reshape(delta.shape[0], -1).mean(axis=1)

    LOGGER.info(
        "Saved %s retrieved data: images=%s normal_scores=%s pathological_scores=%s delta_scores=%s",
        split_name,
        len(targets),
        np.asarray(normal_scores).shape,
        np.asarray(pathological_scores).shape,
        delta_scores.shape,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    if args.batch_size != 1:
        raise SystemExit("Retrieved-data extraction requires --batch-size 1 so features_<idx>.npy matches sample indices.")

    args.num_patches = 256
    require_bank(args.normal_bank, "Normal knowledge bank")
    require_bank(args.pathological_bank, "Pathological knowledge bank")
    require_file(args.backbone_checkpoint, "Backbone checkpoint")
    if not (args.data_root / "fundus").exists():
        raise FileNotFoundError(f"Fundus CSV root not found: {args.data_root / 'fundus'}")

    utils.fix_seeds(args.seed)
    device = utils.set_torch_device([args.gpu] if torch.cuda.is_available() else [])

    normal_bank = load_bank(args.normal_bank, args.backbone_checkpoint, args.nn, device, args.faiss_on_gpu, args.num_workers)
    pathological_bank = load_bank(args.pathological_bank, args.backbone_checkpoint, args.nn, device, args.faiss_on_gpu, args.num_workers)

    for split_name in args.test_sets:
        extract_split(args, split_name, normal_bank, pathological_bank)


if __name__ == "__main__":
    main()
