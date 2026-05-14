#!/usr/bin/env python3
"""Prepare the fundus data splits used by EviScreen.

The script keeps the release workflow intentionally narrow:
BRSET and EDDFS are used for training, and JSIEC_original and RIADD_original are
used for testing. It also applies the same circular fundus crop used in the
original experiments for BRSET, EDDFS, and RIADD.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess_fundus_images import preprocess_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare EviScreen fundus data.")
    parser.add_argument("--raw-root", required=True, type=Path, help="Root containing raw BRSET, EDDFS, RIADD, and JSIEC folders.")
    parser.add_argument("--output-root", required=True, type=Path, help="Output root for dataset/fundus CSV files.")
    parser.add_argument("--num-workers", default=8, type=int, help="Image preprocessing worker threads.")
    parser.add_argument("--seed", default=0, type=int, help="Seed for all deterministic splits.")
    parser.add_argument("--skip-preprocess", action="store_true", help="Only generate CSV files.")
    parser.add_argument("--skip-csv", action="store_true", help="Only preprocess images.")
    return parser.parse_args()


def output_dirs(output_root: Path) -> dict[str, Path]:
    root = output_root / "fundus"
    dirs = {
        "BRSET": root / "BRSET",
        "EDDFS": root / "EDDFS",
        "JSIEC": root / "JSIEC",
        "RIADD": root / "RIADD",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def prepare_brset(raw_root: Path, out_dir: Path, seed: int) -> None:
    np.random.seed(seed)
    brset_root = raw_root / "brazilian-ophthalmological" / "1.0.1"
    df = pd.read_csv(brset_root / "labels_brset.csv")
    df = df.copy()
    df = df[df["quality"] == "Adequate"]
    df["abnormal"] = (df.iloc[:, 20:33].sum(axis=1) != 0).astype(int)
    normal = df[df["abnormal"] == 0]
    abnormal_no_other = df[(df["abnormal"] != 0) & (df["other"] == 0)]
    df = pd.concat([normal, abnormal_no_other], ignore_index=True)

    patient_ids = df["patient_id"].unique()
    test_patient_ids = np.random.choice(patient_ids, size=500, replace=False)
    remaining_patient_ids = np.setdiff1d(patient_ids, test_patient_ids)
    val_patient_ids = np.random.choice(remaining_patient_ids, size=200, replace=False)
    train_patient_ids = np.setdiff1d(remaining_patient_ids, val_patient_ids)

    train = df[df["patient_id"].isin(train_patient_ids)]
    val = df[df["patient_id"].isin(val_patient_ids)]
    test = df[df["patient_id"].isin(test_patient_ids)]

    train.to_csv(out_dir / "train_original.csv", index=False)
    val.to_csv(out_dir / "val_original.csv", index=False)
    test.to_csv(out_dir / "test_original.csv", index=False)


def prepare_eddfs(raw_root: Path, out_dir: Path, seed: int) -> None:
    eddfs_root = raw_root / "EDDFS"
    train = pd.read_csv(eddfs_root / "train.csv")
    test = pd.read_csv(eddfs_root / "test.csv")

    def remove_others(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["abnormal"] = df["normal"].apply(lambda x: 0 if x == 1 else 1)
        normal = df[df["normal"] == 1]
        abnormal_no_other = df[(df["normal"] == 0) & (df["others"] == 0)]
        return pd.concat([normal, abnormal_no_other], ignore_index=True)

    train = remove_others(train)
    test = remove_others(test)
    val = train.sample(n=400, random_state=seed)
    train = train.drop(val.index)
    val = pd.concat([val[val["abnormal"] == 0], val[val["abnormal"] == 1]], ignore_index=True)

    train.to_csv(out_dir / "train_original.csv", index=False)
    val.to_csv(out_dir / "val_original.csv", index=False)
    test.to_csv(out_dir / "test_original.csv", index=False)


def prepare_train_5000(brset_dir: Path, eddfs_dir: Path, seed: int) -> None:
    brset = pd.read_csv(brset_dir / "train_original.csv")
    eddfs = pd.read_csv(eddfs_dir / "train_original.csv")

    brset_5000 = pd.concat(
        [
            brset[brset["abnormal"] == 0].sample(n=2500, random_state=seed),
            brset[brset["abnormal"] == 1].sample(n=2500, random_state=seed),
        ],
        ignore_index=True,
    )
    eddfs_5000 = pd.concat(
        [
            eddfs[eddfs["abnormal"] == 0].sample(n=2500, random_state=seed),
            eddfs[eddfs["abnormal"] == 1].sample(n=2500, random_state=seed),
        ],
        ignore_index=True,
    )
    brset_5000.to_csv(brset_dir / "train_original_for_5000.csv", index=False)
    eddfs_5000.to_csv(eddfs_dir / "train_original_for_5000.csv", index=False)
    prepare_train_5000_remain(brset_dir, eddfs_dir)


def prepare_train_5000_remain(brset_dir: Path, eddfs_dir: Path) -> None:
    brset = pd.read_csv(brset_dir / "train_original.csv")
    eddfs = pd.read_csv(eddfs_dir / "train_original.csv")
    brset_5000 = pd.read_csv(brset_dir / "train_original_for_5000.csv")
    eddfs_5000 = pd.read_csv(eddfs_dir / "train_original_for_5000.csv")

    brset_remain = brset[~brset["image_id"].astype(str).isin(set(brset_5000["image_id"].astype(str)))]
    eddfs_remain = eddfs[~eddfs["fnames"].astype(str).isin(set(eddfs_5000["fnames"].astype(str)))]
    brset_remain.to_csv(brset_dir / "train_original_5000_remain.csv", index=False)
    eddfs_remain.to_csv(eddfs_dir / "train_original_5000_remain.csv", index=False)


def prepare_jsiec(raw_root: Path, out_dir: Path) -> None:
    root = raw_root / "JSIEC" / "1000images"
    rows = []
    for folder in sorted(os.listdir(root)):
        folder_path = root / folder
        if not folder_path.is_dir():
            continue
        for fname in os.listdir(folder_path):
            rows.append(
                {
                    "fnames": fname,
                    "dirs": folder,
                    "labels": folder,
                    "abnormal": 0 if folder == "0.0.Normal" else 1,
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "test_original.csv", index=False)


def prepare_riadd(raw_root: Path, out_dir: Path) -> None:
    root = raw_root / "RIADD"
    splits = [
        (root / "train_set" / "RFMiD_Training_Labels.csv", "train_set/Training_Preprocessed"),
        (root / "val_set" / "RFMiD_Validation_Label.csv", "val_set/Validation_Preprocessed"),
        (root / "test_set" / "RFMiD_Testing_Labels.csv", "test_set/Test_Preprocessed"),
    ]
    frames = []
    for csv_path, image_dir in splits:
        df = pd.read_csv(csv_path).copy()
        df["dir"] = image_dir
        df["abnormal"] = df["Disease_Risk"].apply(lambda x: 0 if x == 0 else 1)
        frames.append(df)
    pd.concat(frames, ignore_index=True).to_csv(out_dir / "test_original.csv", index=False)


def validate_5000_split(path: Path, id_column: str) -> None:
    df = pd.read_csv(path)
    counts = df["abnormal"].value_counts().to_dict()
    if len(df) != 5000 or counts.get(0, 0) != 2500 or counts.get(1, 0) != 2500:
        raise ValueError(f"{path} must contain 2500 normal and 2500 abnormal rows, got {counts} with {len(df)} rows")
    if df[id_column].duplicated().any():
        raise ValueError(f"{path} contains duplicate {id_column} entries")


def validate_remain_split(dataset_dir: Path, id_column: str) -> None:
    train = pd.read_csv(dataset_dir / "train_original.csv")
    train_5000 = pd.read_csv(dataset_dir / "train_original_for_5000.csv")
    remain = pd.read_csv(dataset_dir / "train_original_5000_remain.csv")

    train_ids = set(train[id_column].astype(str))
    train_5000_ids = set(train_5000[id_column].astype(str))
    remain_ids = set(remain[id_column].astype(str))
    if train_5000_ids & remain_ids:
        raise ValueError(f"{dataset_dir} 5000 and remain splits overlap.")
    if train_5000_ids | remain_ids != train_ids:
        raise ValueError(f"{dataset_dir} 5000 and remain splits do not cover train_original.csv.")


def prepare_csvs(raw_root: Path, output_root: Path, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    dirs = output_dirs(output_root)
    prepare_brset(raw_root, dirs["BRSET"], seed)
    prepare_eddfs(raw_root, dirs["EDDFS"], seed)
    prepare_train_5000(dirs["BRSET"], dirs["EDDFS"], seed)
    prepare_jsiec(raw_root, dirs["JSIEC"])
    prepare_riadd(raw_root, dirs["RIADD"])
    validate_5000_split(dirs["BRSET"] / "train_original_for_5000.csv", "image_id")
    validate_5000_split(dirs["EDDFS"] / "train_original_for_5000.csv", "fnames")
    validate_remain_split(dirs["BRSET"], "image_id")
    validate_remain_split(dirs["EDDFS"], "fnames")


def main() -> None:
    args = parse_args()
    if not args.skip_preprocess:
        preprocess_images(args.raw_root, args.num_workers)
    if not args.skip_csv:
        prepare_csvs(args.raw_root, args.output_root, args.seed)


if __name__ == "__main__":
    main()
