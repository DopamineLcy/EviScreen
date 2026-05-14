# Stage 1: Dual Knowledge Bank Construction

This stage trains two fundus knowledge banks:

- `normal`: BRSET+EDDFS normal images from `train_original_for_5000.csv`
- `pathological`: BRSET+EDDFS abnormal images from `train_original_for_5000.csv`

The default model follows the paper workflow: ViT-L DINOv2-MEH, `blocks.7` and `blocks.17`, image size 224, patch size 3, coreset ratio 0.1, and chunk size 5000.

## Train

```bash
python train_dual_knowledge_banks.py \
  --data-root /path/to/dataset \
  --raw-root /path/to/raw_dataset \
  --checkpoint /path/to/RETFound_dinov2_meh.pth \
  --output-dir /path/to/knowledge_banks \
  --gpu 0 \
  --faiss-on-gpu
```

For the local release workflow, run:

```bash
bash build_dual_knowledge_banks.sh
```

This reads CSV files from `../stage_0_data_preparation/fundus_csv` and saves the dual knowledge banks under `experiments/`. The default paths are defined in `../constants.sh`; edit that file or override `DATA_ROOT`, `RAW_ROOT`, `CHECKPOINT`, and `OUTPUT_DIR` from the shell environment.

For a full-path smoke test, run:

```bash
bash debug_dual_knowledge_bank_workflow.sh
```

This builds normal/pathological knowledge banks with 50 randomly selected training images per bank, then evaluates on the full JSIEC_original and RIADD_original test sets.

## Evaluate

```bash
python evaluate_dual_knowledge_banks.py \
  --data-root /path/to/dataset \
  --raw-root /path/to/raw_dataset \
  --normal-bank /path/to/knowledge_banks/normal \
  --pathological-bank /path/to/knowledge_banks/pathological \
  --checkpoint /path/to/RETFound_dinov2_meh.pth \
  --output-dir /path/to/eval_output \
  --test-sets JSIEC_original RIADD_original \
  --nn 16
```

For the local release workflow, run:

```bash
bash evaluate_dual_knowledge_banks.sh
```

This evaluates `experiments/normal` and `experiments/pathological` by default, writes results to `experiments/eval`, and uses JSIEC_original plus RIADD_original with `nn=16`. It also calls the shared `../evaluate_category_metrics.py` script to write `category_mean_metrics.json`. Override `BANK_ROOT`, `NORMAL_BANK`, `PATHOLOGICAL_BANK`, `OUTPUT_DIR`, `DATA_ROOT`, `RAW_ROOT`, `CHECKPOINT`, `GPU`, or `NN` from the shell environment when needed.

The evaluation output includes per-image predictions and AUROC metrics.

To recompute category-wise metrics from existing prediction files:

```bash
bash evaluate_category_metrics.sh
```
