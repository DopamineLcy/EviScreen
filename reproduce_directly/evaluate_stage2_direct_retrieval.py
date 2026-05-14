#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
EVISCREEN_ROOT = SCRIPT_DIR.parent
STAGE1_ROOT = EVISCREEN_ROOT / "stage_1_dual_knowledge_bank_construction"
sys.path.insert(0, str(STAGE1_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from eviscreen.knowledge_bank import common  # noqa: E402
from eviscreen.knowledge_bank.datasets.fundus import DatasetSplit, FundusDataset  # noqa: E402
from retrieving_distance_cat_head import MainClassifier as RetrievingDistanceCatHead  # noqa: E402


LOGGER = logging.getLogger(__name__)

SPLITS = {
    "JSIEC_original": DatasetSplit.JSIEC_ORIGINAL,
    "RIADD_original": DatasetSplit.RIADD_ORIGINAL,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run direct EviScreen stage-2 inference from pre-extracted retrieved evidence.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--normal-bank", required=True, type=Path)
    parser.add_argument("--pathological-bank", required=True, type=Path)
    parser.add_argument("--head-checkpoint", required=True, type=Path)
    parser.add_argument("--retrieved-data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--test-sets", nargs="+", default=["JSIEC_original", "RIADD_original"], choices=sorted(SPLITS))
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--resize", default=224, type=int)
    parser.add_argument("--imagesize", default=224, type=int)
    parser.add_argument("--max-test-samples", default=None, type=int)
    parser.add_argument("--embed-dim", default=None, type=int)
    parser.add_argument("--num-patches", default=None, type=int)
    parser.add_argument("--num-heads", default=None, type=int)
    parser.add_argument("--depth", default=None, type=int)
    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def require_bank(path: Path, description: str) -> None:
    require_file(path / "nnscorer_search_index.faiss", f"{description} FAISS index")


def make_patchcore_index_proxy(nn_method):
    return SimpleNamespace(anomaly_scorer=SimpleNamespace(nn_method=nn_method))


def load_patchcore_index_proxies(normal_bank: Path, pathological_bank: Path):
    # Match head_src/main_pretrain.py: one FaissNN instance is passed to both
    # PatchCore.load_from_path calls, so the second load updates the same
    # nn_method object held by both patchcore instances.
    nn_method = common.FaissNN(False, 1)
    patchcore_instance_normal = make_patchcore_index_proxy(nn_method)
    patchcore_instance_normal.anomaly_scorer.nn_method.load(str(normal_bank / "nnscorer_search_index.faiss"))

    patchcore_instance_abnormal = make_patchcore_index_proxy(nn_method)
    patchcore_instance_abnormal.anomaly_scorer.nn_method.load(str(pathological_bank / "nnscorer_search_index.faiss"))
    return patchcore_instance_normal, patchcore_instance_abnormal


def torch_load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception as weights_only_error:
        LOGGER.info("Retrying %s with weights_only=False after: %s", path, weights_only_error)
        return torch.load(path, map_location="cpu", weights_only=False)


def checkpoint_model_args(checkpoint: dict, cli_args: argparse.Namespace) -> dict:
    ckpt_args = checkpoint.get("args")
    values = {
        "embed_dim": 1024,
        "num_patches": 256,
        "num_heads": 8,
        "depth": 4,
    }

    if ckpt_args is not None:
        if isinstance(ckpt_args, dict):
            checkpoint_model_name = ckpt_args.get("model", "RetrievingDistanceCatHead")
        else:
            checkpoint_model_name = getattr(ckpt_args, "model", "RetrievingDistanceCatHead")
        if checkpoint_model_name != "RetrievingDistanceCatHead":
            raise ValueError(f"Expected a RetrievingDistanceCatHead checkpoint, got {checkpoint_model_name}.")
        for key in values:
            if isinstance(ckpt_args, dict):
                values[key] = ckpt_args.get(key, values[key])
            else:
                values[key] = getattr(ckpt_args, key, values[key])

    for key in values:
        override = getattr(cli_args, key)
        if override is not None:
            values[key] = override

    return values


def load_head(checkpoint_path: Path, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    checkpoint = torch_load_checkpoint(checkpoint_path)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Head checkpoint must contain a 'model' state_dict: {checkpoint_path}")

    model_args = checkpoint_model_args(checkpoint, args)
    model = RetrievingDistanceCatHead(**model_args)
    msg = model.load_state_dict(checkpoint["model"], strict=True)
    LOGGER.info("Loaded Stage 2 RetrievingDistanceCatHead from %s with args %s: %s", checkpoint_path, model_args, msg)
    model.to(device)
    model.eval()
    return model


class CachedRetrievedEvidenceDataset(Dataset):
    """Dataset matching the retrieved-data layout generated by prepare_stage2_retrieved_data.py."""

    def __init__(
        self,
        split: str,
        retrieved_data_root: Path,
        patchcore_instance_normal,
        patchcore_instance_abnormal,
        image_paths: list[str],
        expected_labels: np.ndarray,
        limit: int | None = None,
    ):
        super().__init__()
        root = Path(retrieved_data_root)
        self.split = split
        self.normal_dir = root / f"{split}_normal"
        self.pathological_dir = root / f"{split}_abnormal"
        self.patchcore_instance_normal = patchcore_instance_normal
        self.patchcore_instance_abnormal = patchcore_instance_abnormal
        self.image_paths = list(image_paths)
        expected_labels = np.asarray(expected_labels, dtype=np.int64)

        self.distances_from_normal = self._load_array(root / f"{split}_distances.npy", np.float32)
        self.distances_from_pathological = self._load_array(root / f"{split}_abnormal_distances.npy", np.float32)
        self.labels = self._load_array(root / f"{split}_anomaly_labels.npy", np.int64)

        if limit is not None:
            self.image_paths = self.image_paths[:limit]
            expected_labels = expected_labels[:limit]
            self.distances_from_normal = self.distances_from_normal[:limit]
            self.distances_from_pathological = self.distances_from_pathological[:limit]
            self.labels = self.labels[:limit]

        if len(self.labels) != len(self.image_paths):
            raise ValueError(f"{split}: retrieved labels contain {len(self.labels)} rows, but split metadata contains {len(self.image_paths)} rows.")
        if len(expected_labels) != len(self.labels):
            raise ValueError(f"{split}: current split labels contain {len(expected_labels)} rows, but retrieved labels contain {len(self.labels)}.")
        if not np.array_equal(self.labels, expected_labels):
            mismatch_count = int(np.count_nonzero(self.labels != expected_labels))
            raise ValueError(f"{split}: retrieved labels do not match current split label order ({mismatch_count} mismatches).")
        for directory_name, directory in (("normal", self.normal_dir), ("pathological", self.pathological_dir)):
            if not directory.is_dir():
                raise FileNotFoundError(f"{split}: {directory_name} retrieved-data directory not found: {directory}")

    @staticmethod
    def _load_array(path: Path, dtype) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(f"Retrieved-data file not found: {path}")
        return np.load(path).astype(dtype, copy=False)

    def __getitem__(self, index: int):
        cur_features = torch.from_numpy(np.load(self.normal_dir / f"features_{index}.npy")).float()
        cur_query_nns_from_normal_indices = torch.from_numpy(np.load(self.normal_dir / f"query_nns_{index}.npy"))
        cur_query_nns_from_abnormal_indices = torch.from_numpy(np.load(self.pathological_dir / f"query_nns_{index}.npy"))

        indices_flat_normal = cur_query_nns_from_normal_indices.reshape(-1).long().cpu().numpy()
        retrieved_features_flat_normal = self.patchcore_instance_normal.anomaly_scorer.nn_method.search_index.reconstruct_batch(indices_flat_normal)
        cur_retrieved_features_from_normal = torch.from_numpy(retrieved_features_flat_normal).view(
            cur_query_nns_from_normal_indices.shape[0],
            cur_query_nns_from_normal_indices.shape[1],
            -1,
        ).to(torch.float16)

        indices_flat_abnormal = cur_query_nns_from_abnormal_indices.reshape(-1).long().cpu().numpy()
        retrieved_features_flat_abnormal = self.patchcore_instance_abnormal.anomaly_scorer.nn_method.search_index.reconstruct_batch(indices_flat_abnormal)
        cur_retrieved_features_from_pathological = torch.from_numpy(retrieved_features_flat_abnormal).view(
            cur_query_nns_from_abnormal_indices.shape[0],
            cur_query_nns_from_abnormal_indices.shape[1],
            -1,
        ).to(torch.float16)

        cur_distances_from_normal = self.distances_from_normal[index]
        cur_distances_from_pathological = self.distances_from_pathological[index]
        label = self.labels[index]

        return (
            cur_features,
            cur_retrieved_features_from_normal,
            cur_distances_from_normal,
            cur_retrieved_features_from_pathological,
            cur_distances_from_pathological,
            label,
        )

    def __len__(self) -> int:
        return int(self.labels.shape[0])


def split_metadata(args: argparse.Namespace, split_name: str) -> tuple[list[str], np.ndarray]:
    dataset = FundusDataset(
        data_root=args.data_root,
        raw_root=args.raw_root,
        classname="fundus",
        resize=args.resize,
        imagesize=args.imagesize,
        split=SPLITS[split_name],
    )
    image_paths = list(dataset.img_paths)
    targets = np.asarray(dataset.targets, dtype=np.int64)
    if args.max_test_samples is not None:
        limit = min(args.max_test_samples, len(targets))
        image_paths = image_paths[:limit]
        targets = targets[:limit]
    return image_paths, targets


def evaluate_split(
    args: argparse.Namespace,
    split_name: str,
    model: torch.nn.Module,
    patchcore_instance_normal,
    patchcore_instance_abnormal,
    device: torch.device,
) -> dict:
    image_paths, targets = split_metadata(args, split_name)
    dataset = CachedRetrievedEvidenceDataset(
        split=split_name,
        retrieved_data_root=args.retrieved_data_root,
        patchcore_instance_normal=patchcore_instance_normal,
        patchcore_instance_abnormal=patchcore_instance_abnormal,
        image_paths=image_paths,
        expected_labels=targets,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    all_labels = []
    all_scores = []
    autocast_context = torch.cuda.amp.autocast if device.type == "cuda" else contextlib.nullcontext

    with torch.no_grad():
        for batch in loader:
            (
                cur_features,
                cur_retrieved_features_from_normal,
                cur_distances_from_normal,
                cur_retrieved_features_from_pathological,
                cur_distances_from_pathological,
                labels,
            ) = batch

            cur_features = cur_features.to(device, non_blocking=True)
            cur_retrieved_features_from_normal = cur_retrieved_features_from_normal.to(device, non_blocking=True)
            cur_distances_from_normal = cur_distances_from_normal.to(device, non_blocking=True)
            cur_retrieved_features_from_pathological = cur_retrieved_features_from_pathological.to(device, non_blocking=True)
            cur_distances_from_pathological = cur_distances_from_pathological.to(device, non_blocking=True)

            if device.type != "cuda":
                cur_retrieved_features_from_normal = cur_retrieved_features_from_normal.float()
                cur_retrieved_features_from_pathological = cur_retrieved_features_from_pathological.float()

            with autocast_context():
                logits = model(
                    cur_features,
                    cur_retrieved_features_from_normal,
                    cur_distances_from_normal,
                    cur_retrieved_features_from_pathological,
                    cur_distances_from_pathological,
                ).squeeze(-1)
                scores = torch.sigmoid(logits)

            all_labels.append(labels.cpu())
            all_scores.append(scores.float().cpu())

    y_true = torch.cat(all_labels).numpy().astype(np.int64)
    y_score = torch.cat(all_scores).numpy().astype(float)
    metrics = {
        "num_images": int(len(y_true)),
        "auroc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None,
        "ap": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / f"{split_name}_predictions.csv"
    with prediction_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "y_true", "stage2_score"])
        for image_path, label, score in zip(image_paths, y_true, y_score):
            writer.writerow([image_path, int(label), float(score)])

    metrics_path = args.output_dir / f"{split_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    LOGGER.info("%s metrics: %s", split_name, metrics)
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not (args.data_root / "fundus").exists():
        raise FileNotFoundError(f"Fundus CSV root not found: {args.data_root / 'fundus'}")
    require_bank(args.normal_bank, "Normal knowledge bank")
    require_bank(args.pathological_bank, "Pathological knowledge bank")
    require_file(args.head_checkpoint, "Head checkpoint")

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    patchcore_instance_normal, patchcore_instance_abnormal = load_patchcore_index_proxies(args.normal_bank, args.pathological_bank)
    model = load_head(args.head_checkpoint, args, device)

    summary = {
        split: evaluate_split(args, split, model, patchcore_instance_normal, patchcore_instance_abnormal, device)
        for split in args.test_sets
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
