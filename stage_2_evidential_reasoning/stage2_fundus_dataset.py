from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import PIL.Image
import torch
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Stage2FundusDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_root,
        raw_root,
        split: str,
        classname: str = "fundus",
        resize: int = 224,
        imagesize: int = 224,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.raw_root = Path(raw_root)
        self.split = split
        self.classname = classname

        self.eddfs_image_dir = self.raw_root / "EDDFS" / "PreprocessedImages"
        self.brset_image_dir = self.raw_root / "brazilian-ophthalmological" / "1.0.1" / "fundus_photos_preprocessed"
        self.jsiec_image_dir = self.raw_root / "JSIEC" / "1000images"
        self.riadd_image_dir = self.raw_root / "RIADD"

        if split == "fundus_remain_5000":
            self.img_paths, self.targets = self._load_remain_5000()
        elif split == "fundus_val":
            self.img_paths, self.targets = self._load_val()
        elif split == "JSIEC_original":
            self.img_paths, self.targets = self._load_jsiec_original()
        elif split == "RIADD_original":
            self.img_paths, self.targets = self._load_riadd_original()
        else:
            raise ValueError(f"Unsupported stage-2 split: {split}")

        self.labels = ["good" if int(target) == 0 else "bad" for target in self.targets]
        self.data_to_iterate = list(zip(self.img_paths, self.labels))
        self.transform_img = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.imagesize = (3, imagesize, imagesize)

    def _load_remain_5000(self) -> tuple[list[str], np.ndarray]:
        eddfs = pd.read_csv(self.data_root / "fundus" / "EDDFS" / "train_original_5000_remain.csv")
        brset = pd.read_csv(self.data_root / "fundus" / "BRSET" / "train_original_5000_remain.csv")
        return self._eddfs_brset_paths(eddfs, brset, include_brset_test=False)

    def _load_val(self) -> tuple[list[str], np.ndarray]:
        eddfs = pd.read_csv(self.data_root / "fundus" / "EDDFS" / "val_original.csv")
        brset = pd.read_csv(self.data_root / "fundus" / "BRSET" / "val_original.csv")
        brset_test = pd.read_csv(self.data_root / "fundus" / "BRSET" / "test_original.csv")
        return self._eddfs_brset_paths(eddfs, brset, brset_test=brset_test, include_brset_test=True)

    def _eddfs_brset_paths(
        self,
        eddfs: pd.DataFrame,
        brset: pd.DataFrame,
        brset_test: pd.DataFrame | None = None,
        include_brset_test: bool = False,
    ) -> tuple[list[str], np.ndarray]:
        eddfs_normal = eddfs[eddfs["abnormal"] == 0]
        eddfs_abnormal = eddfs[eddfs["abnormal"] == 1]
        brset_normal = brset[brset["abnormal"] == 0]
        brset_abnormal = brset[brset["abnormal"] == 1]

        normal_paths = [self.eddfs_image_dir / fname for fname in eddfs_normal["fnames"].tolist()]
        normal_paths += [self.brset_image_dir / f"{image_id}.jpg" for image_id in brset_normal["image_id"].tolist()]

        abnormal_paths = [self.eddfs_image_dir / fname for fname in eddfs_abnormal["fnames"].tolist()]
        abnormal_paths += [self.brset_image_dir / f"{image_id}.jpg" for image_id in brset_abnormal["image_id"].tolist()]

        if include_brset_test:
            if brset_test is None:
                raise ValueError("brset_test must be provided when include_brset_test=True.")
            brset_test_normal = brset_test[brset_test["abnormal"] == 0]
            brset_test_abnormal = brset_test[brset_test["abnormal"] == 1]
            normal_paths += [self.brset_image_dir / f"{image_id}.jpg" for image_id in brset_test_normal["image_id"].tolist()]
            abnormal_paths += [self.brset_image_dir / f"{image_id}.jpg" for image_id in brset_test_abnormal["image_id"].tolist()]

        paths = [str(path) for path in normal_paths + abnormal_paths]
        targets = np.concatenate((np.zeros(len(normal_paths), dtype=int), np.ones(len(abnormal_paths), dtype=int)))
        return paths, targets

    def _load_jsiec_original(self) -> tuple[list[str], np.ndarray]:
        df = pd.read_csv(self.data_root / "fundus" / "JSIEC" / "test_original.csv")
        return self._jsiec_paths_from_frame(df[df["abnormal"] == 0], df[df["abnormal"] == 1])

    def _jsiec_paths_from_frame(self, normal: pd.DataFrame, abnormal: pd.DataFrame) -> tuple[list[str], np.ndarray]:
        normal_paths = [self.jsiec_image_dir / row["dirs"] / row["fnames"] for _, row in normal.iterrows()]
        abnormal_paths = [self.jsiec_image_dir / row["dirs"] / row["fnames"] for _, row in abnormal.iterrows()]
        paths = [str(path) for path in normal_paths + abnormal_paths]
        targets = np.concatenate((np.zeros(len(normal_paths), dtype=int), np.ones(len(abnormal_paths), dtype=int)))
        return paths, targets

    def _load_riadd_original(self) -> tuple[list[str], np.ndarray]:
        df = pd.read_csv(self.data_root / "fundus" / "RIADD" / "test_original.csv")
        normal = df[df["abnormal"] == 0]
        abnormal = df[df["abnormal"] == 1]
        normal_paths = [self.riadd_image_dir / row["dir"] / f"{self._format_image_id(row['ID'])}.png" for _, row in normal.iterrows()]
        abnormal_paths = [self.riadd_image_dir / row["dir"] / f"{self._format_image_id(row['ID'])}.png" for _, row in abnormal.iterrows()]
        paths = [str(path) for path in normal_paths + abnormal_paths]
        targets = np.concatenate((np.zeros(len(normal_paths), dtype=int), np.ones(len(abnormal_paths), dtype=int)))
        return paths, targets

    @staticmethod
    def _format_image_id(value) -> str:
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    def __getitem__(self, idx: int):
        image_path = self.img_paths[idx]
        anomaly = "good" if int(self.targets[idx]) == 0 else "bad"
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)
        return {
            "image": image,
            "mask": torch.ones([1, *image.size()[1:]]),
            "classname": self.classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": os.path.basename(image_path),
            "image_path": image_path,
        }

    def __len__(self) -> int:
        return len(self.img_paths)
