from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset


SPLIT_TO_DATASET = {
    "train": "fundus_remain_5000",
    "val": "fundus_val",
    "fundus_remain_5000": "fundus_remain_5000",
    "fundus_val": "fundus_val",
    "JSIEC_original": "JSIEC_original",
    "RIADD_original": "RIADD_original",
}


def make_patchcore_index_proxy(nn_method):
    return SimpleNamespace(anomaly_scorer=SimpleNamespace(nn_method=nn_method))


def load_patchcore_index_proxies(common_module, normal_bank: Path, pathological_bank: Path):
    # Match head_src/main_pretrain.py exactly: both PatchCore instances receive
    # the same FaissNN object, and the second load updates that object.
    nn_method = common_module.FaissNN(False, 1)
    patchcore_instance_normal = make_patchcore_index_proxy(nn_method)
    patchcore_instance_normal.anomaly_scorer.nn_method.load(str(normal_bank / "nnscorer_search_index.faiss"))

    patchcore_instance_abnormal = make_patchcore_index_proxy(nn_method)
    patchcore_instance_abnormal.anomaly_scorer.nn_method.load(str(pathological_bank / "nnscorer_search_index.faiss"))
    return patchcore_instance_normal, patchcore_instance_abnormal


class RetrievedDistanceDataset(Dataset):
    """Stage-2 retrieved-distance dataset matching head_src/datasets/RetrivedDistance_dataset.py."""

    def __init__(
        self,
        split: str,
        retrieved_data_root: Path,
        patchcore_instance_normal,
        patchcore_instance_abnormal,
        limit: int | None = None,
    ):
        super().__init__()
        if split not in SPLIT_TO_DATASET:
            raise ValueError(f"Unsupported split: {split}")

        self.split = split
        self.dataset_name = SPLIT_TO_DATASET[split]
        self.root = Path(retrieved_data_root)
        self.normal_directory = self.root / f"{self.dataset_name}_normal"
        self.abnormal_directory = self.root / f"{self.dataset_name}_abnormal"
        self.patchcore_instance_normal = patchcore_instance_normal
        self.patchcore_instance_abnormal = patchcore_instance_abnormal

        self.distances_from_normal = self._load_array(self.root / f"{self.dataset_name}_distances.npy", np.float32)
        self.distances_from_abnormal = self._load_array(self.root / f"{self.dataset_name}_abnormal_distances.npy", np.float32)
        self.labels = self._load_array(self.root / f"{self.dataset_name}_anomaly_labels.npy", np.int64)

        if limit is not None:
            self.distances_from_normal = self.distances_from_normal[:limit]
            self.distances_from_abnormal = self.distances_from_abnormal[:limit]
            self.labels = self.labels[:limit]

        for directory_name, directory in (("normal", self.normal_directory), ("abnormal", self.abnormal_directory)):
            if not directory.is_dir():
                raise FileNotFoundError(f"{self.dataset_name}: {directory_name} retrieved-data directory not found: {directory}")

    @staticmethod
    def _load_array(path: Path, dtype) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(f"Retrieved-data file not found: {path}")
        return np.load(path).astype(dtype, copy=False)

    def __getitem__(self, index: int):
        cur_features = torch.from_numpy(np.load(self.normal_directory / f"features_{index}.npy")).float()
        cur_query_nns_from_normal_indices = torch.from_numpy(np.load(self.normal_directory / f"query_nns_{index}.npy"))
        cur_query_nns_from_abnormal_indices = torch.from_numpy(np.load(self.abnormal_directory / f"query_nns_{index}.npy"))

        indices_flat_normal = cur_query_nns_from_normal_indices.reshape(-1).long().cpu().numpy()
        retrieved_features_flat_normal = self.patchcore_instance_normal.anomaly_scorer.nn_method.search_index.reconstruct_batch(indices_flat_normal)
        cur_retrieved_features_from_normal = torch.from_numpy(retrieved_features_flat_normal).view(
            cur_query_nns_from_normal_indices.shape[0],
            cur_query_nns_from_normal_indices.shape[1],
            -1,
        ).to(torch.float16)

        indices_flat_abnormal = cur_query_nns_from_abnormal_indices.reshape(-1).long().cpu().numpy()
        retrieved_features_flat_abnormal = self.patchcore_instance_abnormal.anomaly_scorer.nn_method.search_index.reconstruct_batch(indices_flat_abnormal)
        cur_retrieved_features_from_abnormal = torch.from_numpy(retrieved_features_flat_abnormal).view(
            cur_query_nns_from_abnormal_indices.shape[0],
            cur_query_nns_from_abnormal_indices.shape[1],
            -1,
        ).to(torch.float16)

        cur_distances_from_normal = self.distances_from_normal[index]
        cur_distances_from_abnormal = self.distances_from_abnormal[index]
        cur_labels = self.labels[index]

        return (
            cur_features,
            cur_retrieved_features_from_normal,
            cur_distances_from_normal,
            cur_retrieved_features_from_abnormal,
            cur_distances_from_abnormal,
            cur_labels,
        )

    def __len__(self) -> int:
        return int(self.labels.shape[0])
