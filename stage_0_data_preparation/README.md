# Stage 0: Fundus Data Preparation

This stage prepares the fundus-only workflow. BRSET and EDDFS are used for training, and JSIEC_original and RIADD_original are used for testing.

## Dataset Downloads

Download the raw datasets manually and place them under one `raw_root`.

| Dataset | Source | Expected location under `raw_root` |
| --- | --- | --- |
| BRSET | https://physionet.org/content/brazilian-ophthalmological/1.0.1/ | `brazilian-ophthalmological/1.0.1/` |
| EDDFS | https://github.com/xia-xx-cv/EDDFS_dataset/ | `EDDFS/` |
| EDDFS annotations | https://github.com/xia-xx-cv/EDDFS_dataset/tree/main/datas/EDDFS/Annotation | `EDDFS/train.csv`, `EDDFS/test.csv` |
| RIADD | https://riadd.grand-challenge.org/Download/ | `RIADD/` |
| JSIEC | https://www.kaggle.com/datasets/linchundan/fundusimage1000 | `JSIEC/` |

The expected directory layout is:

```text
<raw_root>/
  brazilian-ophthalmological/
    1.0.1/
      labels_brset.csv
      fundus_photos/
  EDDFS/
    OriginalImages/
    train.csv
    test.csv
  RIADD/
    train_set/
      Training/
      RFMiD_Training_Labels.csv
    val_set/
      Validation/
      RFMiD_Validation_Label.csv
    test_set/
      Test/
      RFMiD_Testing_Labels.csv
  JSIEC/
    1000images/
```

## Image Preprocessing

BRSET, EDDFS, and RIADD are preprocessed with the same fundus crop used in the original experiments:

1. Read each image with OpenCV.
2. Build a foreground mask using pixels whose channel sum is larger than 30.
3. Run connected-component labeling on the mask.
4. Keep the largest connected component as the fundus region.
5. Build a square crop centered at that component centroid, using the larger side of the component bounding box.
6. Add a 5-pixel margin.
7. Crop the image; if the crop goes outside the original image, pad with black pixels.
8. Resize the crop to `512x512` using bilinear interpolation.
9. Save the preprocessed image next to the raw dataset:
   - BRSET: `fundus_photos/` -> `fundus_photos_preprocessed/`
   - EDDFS: `OriginalImages/` -> `PreprocessedImages/`
   - RIADD: `Training/Validation/Test` -> `Training_Preprocessed/Validation_Preprocessed/Test_Preprocessed`

JSIEC is not separately preprocessed in this workflow; the CSV points to the downloaded `JSIEC/1000images` files directly.

To run image preprocessing only:

```bash
bash preprocess_fundus_images.sh
```

Override the default raw data location and worker count when needed:

```bash
RAW_ROOT=/path/to/raw_dataset NUM_WORKERS=8 bash preprocess_fundus_images.sh
```

To run preprocessing and generate CSV files in one command:

```bash
python prepare_fundus_data.py \
  --raw-root <raw_root> \
  --output-root <dataset_root> \
  --num-workers 8 \
  --seed 0
```

Use `--skip-preprocess` if the preprocessed images already exist and only the CSV files need to be regenerated. Use `--skip-csv` if only image preprocessing is needed.

## CSV Generation

For the local release workflow, run:

```bash
bash generate_fundus_csv.sh
```

By default this uses:

- Python: `EVISCREEN_BANK_PYTHON_BIN` from `../constants.sh`
- Raw root: `EVISCREEN_RAW_ROOT` from `../constants.sh`
- Output root: `stage_0_data_preparation/fundus_csv`
- Seed: `0`

The shell script runs with `--skip-preprocess`, so it assumes the BRSET, EDDFS, and RIADD preprocessed image folders already exist. Edit `../constants.sh` for persistent local defaults, or override paths through environment variables when needed:

```bash
RAW_ROOT=/path/to/raw_dataset OUTPUT_ROOT=/path/to/fundus_csv bash generate_fundus_csv.sh
```

Generated CSV files:

- BRSET: `train_original.csv`, `val_original.csv`, `test_original.csv`, `train_original_for_5000.csv`, `train_original_5000_remain.csv`
- EDDFS: `train_original.csv`, `val_original.csv`, `test_original.csv`, `train_original_for_5000.csv`, `train_original_5000_remain.csv`
- JSIEC: `test_original.csv`
- RIADD: `test_original.csv`

For BRSET and EDDFS, `train_original_5000_remain.csv` is generated as
`train_original.csv - train_original_for_5000.csv`. The script validates that
the two training subsets do not overlap and that their union equals
`train_original.csv`. Stage 2 uses this remain split as `fundus_remain_5000`
for evidential head training.

The local workflow writes these files under:

```text
stage_0_data_preparation/fundus_csv/fundus/
```

Use `stage_0_data_preparation/fundus_csv` as stage 1 `--data-root`. This is also the default `DATA_ROOT` used by `../stage_1_dual_knowledge_bank_construction/build_dual_knowledge_banks.sh`.
