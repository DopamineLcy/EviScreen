#!/usr/bin/env python3
"""Preprocess BRSET, EDDFS, and RIADD fundus images.

This reproduces the fundus crop used in the original experiment setup under
dataset_establishment/fundus. JSIEC is not preprocessed in this workflow.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from operator import attrgetter
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from skimage.measure import label, regionprops
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess EviScreen fundus images.")
    parser.add_argument("--raw-root", required=True, type=Path, help="Root containing raw BRSET, EDDFS, and RIADD folders.")
    parser.add_argument("--num-workers", default=8, type=int, help="Image preprocessing worker threads.")
    return parser.parse_args()


def fill_crop(img: np.ndarray, min_idx: list[int], max_idx: list[int]) -> np.ndarray:
    crop = np.zeros(np.array(max_idx, dtype="int16") - np.array(min_idx, dtype="int16"), dtype=img.dtype)
    img_shape = np.array(img.shape)
    start = np.array(min_idx, dtype="int16")
    crop_shape = np.array(crop.shape)
    end = start + crop_shape

    crop_low = np.clip(0 - start, a_min=0, a_max=crop_shape)
    crop_high = crop_shape - np.clip(end - img_shape, a_min=0, a_max=crop_shape)
    crop_slices = tuple(slice(low, high) for low, high in zip(crop_low, crop_high))

    pos = np.clip(start, a_min=0, a_max=img_shape)
    end = np.clip(end, a_min=0, a_max=img_shape)
    img_slices = tuple(slice(low, high) for low, high in zip(pos, end))
    crop[crop_slices] = img[img_slices]
    return crop


def fundus_crop(image: np.ndarray, shape: tuple[int, int] = (512, 512), margin: int = 5) -> np.ndarray:
    mask = label(image.sum(axis=-1) > 30)
    regions = regionprops(mask)
    if not regions:
        raise ValueError("No fundus foreground region detected.")
    region = max(regions, key=attrgetter("area"))
    length = (np.array(region.bbox[2:4]) - np.array(region.bbox[0:2])).max()
    bbox = np.concatenate([np.array(region.centroid) - length / 2, np.array(region.centroid) + length / 2]).astype("int16")
    image_b = fill_crop(image, [bbox[0] - margin, bbox[1] - margin, 0], [bbox[2] + margin, bbox[3] + margin, 3])
    return cv2.resize(image_b, shape, interpolation=cv2.INTER_LINEAR)


def preprocess_files(files: list[Path], save_path_fn: Callable[[Path], Path], num_workers: int) -> None:
    def process_one(path: Path) -> None:
        try:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError("cv2.imread returned None")
            image_crop = fundus_crop(image)
            save_path = save_path_fn(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), image_crop)
        except Exception as exc:
            print(f"[warn] failed to preprocess {path}: {exc}")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(process_one, files), total=len(files), desc="preprocess"))


def preprocess_brset(raw_root: Path, num_workers: int) -> None:
    brset_root = raw_root / "brazilian-ophthalmological" / "1.0.1"
    files = [Path(p) for p in glob(str(brset_root / "fundus_photos" / "*.jpg"))]
    preprocess_files(files, lambda p: Path(str(p).replace("fundus_photos", "fundus_photos_preprocessed")), num_workers)


def preprocess_eddfs(raw_root: Path, num_workers: int) -> None:
    eddfs_root = raw_root / "EDDFS"
    files = [Path(p) for p in glob(str(eddfs_root / "OriginalImages" / "*")) if Path(p).is_file()]
    preprocess_files(files, lambda p: Path(str(p).replace("OriginalImages", "PreprocessedImages")), num_workers)


def preprocess_riadd(raw_root: Path, num_workers: int) -> None:
    riadd_root = raw_root / "RIADD"
    files = (
        [Path(p) for p in glob(str(riadd_root / "train_set" / "Training" / "*.png"))]
        + [Path(p) for p in glob(str(riadd_root / "val_set" / "Validation" / "*.png"))]
        + [Path(p) for p in glob(str(riadd_root / "test_set" / "Test" / "*.png"))]
    )
    preprocess_files(
        files,
        lambda p: Path(str(p).replace("Training", "Training_Preprocessed").replace("Validation", "Validation_Preprocessed").replace("Test", "Test_Preprocessed")),
        num_workers,
    )


def preprocess_images(raw_root: Path, num_workers: int) -> None:
    preprocess_brset(raw_root, num_workers)
    preprocess_eddfs(raw_root, num_workers)
    preprocess_riadd(raw_root, num_workers)


def main() -> None:
    args = parse_args()
    preprocess_images(args.raw_root, args.num_workers)


if __name__ == "__main__":
    main()
