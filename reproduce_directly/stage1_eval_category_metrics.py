#!/usr/bin/env python3
from __future__ import annotations

"""Compatibility wrapper for the shared EviScreen category-metrics evaluator."""

from pathlib import Path
import runpy

EVISCREEN_ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(EVISCREEN_ROOT / "evaluate_category_metrics.py"), run_name="__main__")
raise SystemExit

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
EVISCREEN_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = EVISCREEN_ROOT.parent
DEFAULT_EVAL_DIR = SCRIPT_DIR / "eval"

METRIC_NAMES = ["AUROC", "AP", "Pat95R", "Pat99R", "Pat100R", "CSR"]

DATASET_CONFIGS = {
    "JSIEC_original": {
        "csv_path": PROJECT_ROOT / "dataset" / "fundus" / "JSIEC" / "test_original.csv",
        "path_token": "1000images",
        "categories": [
            "0.1.Tessellated fundus",
            "0.2.Large optic cup",
            "0.3.DR1",
            "1.0.DR2",
            "1.1.DR3",
            "10.0.Possible glaucoma",
            "10.1.Optic atrophy",
            "11.Severe hypertensive retinopathy",
            "12.Disc swelling and elevation",
            "13.Dragged Disc",
            "14.Congenital disc abnormality",
            "15.0.Retinitis pigmentosa",
            "15.1.Bietti crystalline dystrophy",
            "16.Peripheral retinal degeneration and break",
            "17.Myelinated nerve fiber",
            "18.Vitreous particles",
            "19.Fundus neoplasm",
            "2.0.BRVO",
            "2.1.CRVO",
            "20.Massive hard exudates",
            "21.Yellow-white spots-flecks",
            "22.Cotton-wool spots",
            "23.Vessel tortuosity",
            "24.Chorioretinal atrophy-coloboma",
            "25.Preretinal hemorrhage",
            "26.Fibrosis",
            "27.Laser Spots",
            "28.Silicon oil in eye",
            "29.0.Blur fundus without PDR",
            "29.1.Blur fundus with suspected PDR",
            "3.RAO",
            "4.Rhegmatogenous RD",
            "5.0.CSCR",
            "5.1.VKH disease",
            "6.Maculopathy",
            "7.ERM",
            "8.MH",
            "9.Pathological myopia",
        ],
    },
    "RIADD_original": {
        "csv_path": PROJECT_ROOT / "dataset" / "fundus" / "RIADD" / "test_original.csv",
        "path_token": "RIADD",
        "categories": [
            "DR",
            "ARMD",
            "MH",
            "DN",
            "MYA",
            "BRVO",
            "TSLN",
            "ERM",
            "LS",
            "MS",
            "CSR",
            "ODC",
            "CRVO",
            "TV",
            "AH",
            "ODP",
            "ODE",
            "ST",
            "AION",
            "PT",
            "RT",
            "RS",
            "CRS",
            "EDN",
            "RPEC",
            "MHL",
            "RP",
            "OTHER",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize EviScreen stage-1 eval outputs with category-wise metrics.")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR, help="Directory containing *_predictions.csv files.")
    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help="Single *_results.txt file with y_true in the first column and score in the last column.",
    )
    parser.add_argument(
        "--split-name",
        choices=sorted(DATASET_CONFIGS),
        default=None,
        help="Dataset split name for --results-file. If omitted, it is inferred from the filename.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to the summary JSON file. Defaults to <eval-dir>/category_mean_metrics.json.",
    )
    return parser.parse_args()


def sat_score(y_true, y_prob, recall_level):
    if not 0 <= recall_level <= 1:
        raise ValueError("recall_level must be between 0 and 1.")

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    if len(tpr) < 2:
        return 0.0

    closest_tpr_idx = np.argmin(np.abs(tpr - recall_level))

    if recall_level == 1.0:
        indices_at_max_tpr = np.where(tpr >= 1.0)[0]
        if len(indices_at_max_tpr) > 0:
            closest_tpr_idx = indices_at_max_tpr[np.argmin(fpr[indices_at_max_tpr])]

    return 1 - fpr[closest_tpr_idx]


def calculate_coverage_score(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    pos_mask = y_true == 1
    neg_mask = y_true == 0

    if not np.any(pos_mask) or not np.any(neg_mask):
        return 0.0

    min_pos_score = np.min(y_prob[pos_mask])
    max_neg_score = np.max(y_prob[neg_mask])

    if min_pos_score > max_neg_score:
        return 1.0

    certain_samples = np.sum((y_prob > max_neg_score) | (y_prob < min_pos_score))
    return float(certain_samples / len(y_true))


def roc_curve(y_true, y_prob):
    from sklearn.metrics import roc_curve as _roc_curve

    return _roc_curve(y_true, y_prob)


def format_image_id(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def prediction_relative_key(image_path: str, token: str) -> str:
    parts = Path(image_path).parts
    if token not in parts:
        joined = str(image_path).replace("\\", "/")
        marker = f"/{token}/"
        if marker in joined:
            return joined.split(marker, 1)[1]
        raise ValueError(f"Cannot locate token '{token}' in image path: {image_path}")
    idx = parts.index(token)
    rel_parts = parts[idx + 1 :]
    if not rel_parts:
        raise ValueError(f"Image path does not contain a suffix after token '{token}': {image_path}")
    return "/".join(rel_parts)


def load_predictions(split_name: str, pred_path: Path) -> tuple[pd.DataFrame, str]:
    config = DATASET_CONFIGS[split_name]
    pred_df = pd.read_csv(pred_path)

    required_columns = {"image_path", "y_true"}
    missing_columns = required_columns - set(pred_df.columns)
    if missing_columns:
        raise ValueError(f"{pred_path} is missing columns: {sorted(missing_columns)}")

    score_source_column = pred_df.columns[-1]
    pred_df["rel_key"] = pred_df["image_path"].map(lambda p: prediction_relative_key(p, config["path_token"]))
    pred_df["y_true"] = pd.to_numeric(pred_df["y_true"], errors="raise").astype(int)
    pred_df["score"] = pd.to_numeric(pred_df[score_source_column], errors="raise")
    return pred_df, score_source_column


def load_labels(split_name: str) -> pd.DataFrame:
    config = DATASET_CONFIGS[split_name]
    df = pd.read_csv(config["csv_path"])

    if split_name == "JSIEC_original":
        df["rel_key"] = df["dirs"].astype(str).str.strip("/") + "/" + df["fnames"].astype(str)
    elif split_name == "RIADD_original":
        df["rel_key"] = df["dir"].astype(str).str.strip("/") + "/" + df["ID"].map(format_image_id) + ".png"
        df["_single_label_abnormal"] = df[config["categories"]].sum(axis=1) == 1
    else:
        raise ValueError(f"Unsupported split: {split_name}")

    df["abnormal"] = pd.to_numeric(df["abnormal"], errors="raise").astype(int)
    return df


def merge_predictions_with_labels(split_name: str, pred_path: Path) -> tuple[pd.DataFrame, str]:
    pred_df, score_source_column = load_predictions(split_name, pred_path)
    label_df = load_labels(split_name)

    merged = pred_df.merge(label_df, on="rel_key", how="inner", validate="one_to_one", suffixes=("", "_label"))
    if len(merged) != len(pred_df):
        pred_keys = set(pred_df["rel_key"])
        label_keys = set(label_df["rel_key"])
        missing_in_labels = sorted(pred_keys - label_keys)[:5]
        missing_in_preds = sorted(label_keys - pred_keys)[:5]
        raise RuntimeError(
            f"{pred_path} could not be aligned with {DATASET_CONFIGS[split_name]['csv_path']}. "
            f"Matched {len(merged)}/{len(pred_df)} rows. "
            f"Missing in labels: {missing_in_labels}. Missing in predictions: {missing_in_preds}."
        )

    if not np.array_equal(merged["y_true"].to_numpy(dtype=int), merged["abnormal"].to_numpy(dtype=int)):
        raise RuntimeError(f"Label mismatch detected after merging {pred_path.name}.")

    return merged, score_source_column


def infer_split_name_from_results_file(results_path: Path) -> str | None:
    stem = results_path.stem
    for split_name in DATASET_CONFIGS:
        if stem == f"{split_name}_results" or stem.startswith(f"{split_name}_results"):
            return split_name
    return None


def load_results_file(split_name: str, results_path: Path) -> tuple[pd.DataFrame, str]:
    label_df = load_labels(split_name)
    results_df = pd.read_csv(results_path, sep=r"\s+", header=None, engine="python")

    if results_df.shape[1] < 2:
        raise ValueError(f"{results_path} must contain at least two columns.")
    if len(results_df) != len(label_df):
        raise RuntimeError(
            f"Row count mismatch for {results_path} and {DATASET_CONFIGS[split_name]['csv_path']}: "
            f"{len(results_df)} vs {len(label_df)}."
        )

    y_true = pd.to_numeric(results_df.iloc[:, 0], errors="raise").astype(int).to_numpy()
    score_source_column = results_df.columns[-1]
    score = pd.to_numeric(results_df.iloc[:, -1], errors="raise").to_numpy()

    if not np.array_equal(y_true, label_df["abnormal"].to_numpy(dtype=int)):
        raise RuntimeError(f"Label order mismatch detected in {results_path}.")

    frame = label_df.copy()
    frame["y_true"] = y_true
    frame["score"] = score
    return frame, str(score_source_column)


def compute_metrics(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_true.size == 0:
        return {name: None for name in METRIC_NAMES}

    if len(np.unique(y_true)) < 2:
        return {
            "AUROC": None,
            "AP": None,
            "Pat95R": 0.0,
            "Pat99R": 0.0,
            "Pat100R": 0.0,
            "CSR": 0.0,
        }

    return {
        "AUROC": float(roc_auc_score(y_true, y_prob)),
        "AP": float(average_precision_score(y_true, y_prob)),
        "Pat95R": float(sat_score(y_true, y_prob, 0.95)),
        "Pat99R": float(sat_score(y_true, y_prob, 0.99)),
        "Pat100R": float(sat_score(y_true, y_prob, 1.0)),
        "CSR": float(calculate_coverage_score(y_true, y_prob)),
    }


def mean_metrics(metric_rows):
    summary = {}
    for metric_name in METRIC_NAMES:
        values = [row[metric_name] for row in metric_rows if row.get(metric_name) is not None]
        summary[metric_name] = float(np.mean(values)) if values else None
    return summary


def format_metric_value(value):
    return "N/A" if value is None else f"{value:.6f}"


def evaluate_split(split_name: str, pred_path: Path) -> dict:
    merged, score_source_column = merge_predictions_with_labels(split_name, pred_path)
    return evaluate_frame(split_name, merged, score_source_column)


def evaluate_frame(split_name: str, frame: pd.DataFrame, score_source_column: str) -> dict:
    config = DATASET_CONFIGS[split_name]

    overall = compute_metrics(frame["y_true"], frame["score"])
    per_category = {}

    if split_name == "JSIEC_original":
        for category in config["categories"]:
            subset = frame[(frame["abnormal"] == 0) | (frame["labels"] == category)]
            if subset.empty:
                continue
            per_category[category] = compute_metrics(subset["y_true"], subset["score"])
    elif split_name == "RIADD_original":
        single_label_mask = frame["_single_label_abnormal"]
        for category in config["categories"]:
            subset = frame[
                (frame["abnormal"] == 0)
                | ((frame["abnormal"] == 1) & single_label_mask & (frame[category] == 1))
            ]
            if subset.empty:
                continue
            per_category[category] = compute_metrics(subset["y_true"], subset["score"])
    else:
        raise ValueError(f"Unsupported split: {split_name}")

    category_mean = mean_metrics(list(per_category.values()))

    return {
        "split": split_name,
        "score_source": "last_column",
        "score_column": score_source_column,
        "num_images": int(len(frame)),
        "overall": overall,
        "category_mean": category_mean,
        "valid_category_count": int(len(per_category)),
        "total_category_count": int(len(config["categories"])),
        "per_category": per_category,
    }


def main() -> None:
    args = parse_args()
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "score_source": "last_column",
        "splits": {},
    }

    if args.results_file is not None:
        results_file = args.results_file
        if not results_file.exists():
            raise SystemExit(f"Results file not found: {results_file}")

        split_name = args.split_name or infer_split_name_from_results_file(results_file)
        if split_name is None:
            raise SystemExit("Could not infer split name from results filename. Pass --split-name explicitly.")

        frame, score_source_column = load_results_file(split_name, results_file)
        split_result = evaluate_frame(split_name, frame, score_source_column)
        summary["results_file"] = str(results_file)
        summary["splits"][split_name] = split_result

        mean_metrics_values = split_result["category_mean"]
        print(
            f"{split_name} | categories={split_result['valid_category_count']}/{split_result['total_category_count']} | "
            + " | ".join(f"{name}={format_metric_value(mean_metrics_values[name])}" for name in METRIC_NAMES)
        )
        output_json = args.output_json or results_file.with_name(f"{results_file.stem}_category_metrics.json")
        output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(f"Saved summary to {output_json}")
        return

    eval_dir = args.eval_dir
    if not eval_dir.exists():
        raise SystemExit(f"Eval directory not found: {eval_dir}")

    pred_files = sorted(eval_dir.glob("*_predictions.csv"))
    if not pred_files:
        raise SystemExit(f"No *_predictions.csv files found in {eval_dir}")

    summary["eval_dir"] = str(eval_dir)

    for pred_path in pred_files:
        split_name = pred_path.name[: -len("_predictions.csv")]
        if split_name not in DATASET_CONFIGS:
            print(f"Skipping unknown split file: {pred_path.name}")
            continue

        split_result = evaluate_split(split_name, pred_path)
        summary["splits"][split_name] = split_result

        mean_metrics_values = split_result["category_mean"]
        print(
            f"{split_name} | categories={split_result['valid_category_count']}/{split_result['total_category_count']} | "
            + " | ".join(f"{name}={format_metric_value(mean_metrics_values[name])}" for name in METRIC_NAMES)
        )

    if not summary["splits"]:
        raise SystemExit("No supported prediction files were processed.")

    output_json = args.output_json or (eval_dir / "category_mean_metrics.json")
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved summary to {output_json}")


if __name__ == "__main__":
    main()
