from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import PIL.Image
import torch
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    JSIEC_ORIGINAL = "JSIEC_original"
    RIADD_ORIGINAL = "RIADD_original"


class FundusDataset(torch.utils.data.Dataset):
    """Fundus dataset for EviScreen dual knowledge bank construction."""

    def __init__(
        self,
        data_root,
        raw_root,
        classname="fundus",
        resize=224,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_scale=5000,
        category="normal",
        **kwargs,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.raw_root = Path(raw_root)
        self.classname = classname
        self.split = split
        self.category = category
        self.transform_std = IMAGENET_STD
        self.transform_mean = IMAGENET_MEAN

        self.brset_image_dir = self.raw_root / "brazilian-ophthalmological" / "1.0.1" / "fundus_photos_preprocessed"
        self.eddfs_image_dir = self.raw_root / "EDDFS" / "PreprocessedImages"
        self.riadd_image_dir = self.raw_root / "RIADD"
        self.jsiec_image_dir = self.raw_root / "JSIEC" / "1000images"

        if split == DatasetSplit.TRAIN:
            self.img_paths, self.targets = self._load_train(train_scale, category)
        elif split == DatasetSplit.JSIEC_ORIGINAL:
            self.img_paths, self.targets = self._load_jsiec_original()
        elif split == DatasetSplit.RIADD_ORIGINAL:
            self.img_paths, self.targets = self._load_riadd_original()
        else:
            raise ValueError(f"Unsupported split: {split}")

        self.labels = ["good" if target == 0 else "bad" for target in self.targets]
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

    def _load_train(self, train_scale, category):
        if category not in {"normal", "abnormal"}:
            raise ValueError("--category must be either 'normal' or 'abnormal'.")

        eddfs = pd.read_csv(self.data_root / "fundus" / "EDDFS" / f"train_original_for_{train_scale}.csv")
        brset = pd.read_csv(self.data_root / "fundus" / "BRSET" / f"train_original_for_{train_scale}.csv")
        target_value = 0 if category == "normal" else 1
        eddfs = eddfs[eddfs["abnormal"] == target_value]
        brset = brset[brset["abnormal"] == target_value]

        eddfs_paths = [self.eddfs_image_dir / fname for fname in eddfs["fnames"].tolist()]
        brset_paths = [self.brset_image_dir / f"{image_id}.jpg" for image_id in brset["image_id"].tolist()]
        img_paths = [str(path) for path in eddfs_paths + brset_paths]
        targets = np.zeros(len(img_paths), dtype=int)
        return img_paths, targets

    def _load_jsiec_original(self):
        df = pd.read_csv(self.data_root / "fundus" / "JSIEC" / "test_original.csv")
        img_paths = [self.jsiec_image_dir / row["dirs"] / row["fnames"] for _, row in df.iterrows()]
        return [str(path) for path in img_paths], df["abnormal"].to_numpy(dtype=int)

    def _load_riadd_original(self):
        df = pd.read_csv(self.data_root / "fundus" / "RIADD" / "test_original.csv")
        img_paths = [self.riadd_image_dir / row["dir"] / f"{self._format_image_id(row['ID'])}.png" for _, row in df.iterrows()]
        return [str(path) for path in img_paths], df["abnormal"].to_numpy(dtype=int)

    @staticmethod
    def _format_image_id(value):
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        anomaly = "good" if self.targets[idx] == 0 else "bad"
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

    def __len__(self):
        return len(self.img_paths)
