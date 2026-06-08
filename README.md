# Allocation Split Expert Streamlit App

This is a flat Streamlit app for filling the `Final Alloc.` column from an uploaded allocation workbook or CSV.

## Files included

- `app.py` — Streamlit UI and file handling
- `allocation_split_numpy_core.py` — NumPy-only model inference and feature engineering
- `model_config.json` — model configuration and feature metadata
- `allocate_group_classifier.npz`
- `allocate_flm_regressor.npz`
- `review_group_classifier.npz`
- `review_flm_regressor.npz`
- `requirements.txt`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Upload format

Supported uploads:

- `.xlsx`
- `.xlsm`
- `.xls`
- `.xlsb`
- `.csv`

The app auto-detects the working table header row for Excel files.

## Feature restriction

The model only uses these columns for prediction:

- Class Name
- Line Name
- Site
- MIL
- FLM
- Cost
- L30
- D30
- D60
- LW
- TTM
- Supply
- Dc Avail
- Rank
- Proj. Demand
- Alloc. Rec.
- Flag

`Final Alloc.` is filled in the exported CSV but is not used as an input feature.

## Updated Model Version

This package has been updated with the larger v3 Allocation Split Expert model artifacts trained with:

- Classifier layers: 512 → 256 → 128
- Regressor layers: 768 → 384 → 192 → 96
- Allocate threshold: 0.40
- Review threshold: 0.62
- Review ranking weights: 40% demand pressure, 60% model probability

The app still uses NumPy-only model inference and the same approved feature columns.
