from __future__ import annotations

import io
import os
import re
import zipfile
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import streamlit as st

from allocation_split_numpy_core import (
    ALLOWED_FEATURES,
    TARGET_COL,
    CANONICAL_ALIASES,
    _norm_name,
    canonicalize_columns,
    load_artifacts,
    predict_dataframe,
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = APP_DIR
WORKING_TABLE_HINTS = ["3.3 Working Table", "Working Table", "working table"]

st.set_page_config(
    page_title="Allocation Split Expert",
    page_icon="📦",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def canonical_label(col: Any) -> Optional[str]:
    return CANONICAL_ALIASES.get(_norm_name(col))


def detect_header_row(raw: pd.DataFrame, min_hits: int = 8) -> int:
    """Detect the worksheet header row using only the approved feature/target names."""
    wanted = set(ALLOWED_FEATURES + [TARGET_COL])
    best_i, best_hits = 0, -1
    search_rows = min(len(raw), 50)
    for i in range(search_rows):
        vals = raw.iloc[i].tolist()
        hits = sum(1 for v in vals if canonical_label(v) in wanted)
        if hits > best_hits:
            best_i, best_hits = i, hits
    if best_hits < min_hits:
        # CSV files usually already have headers; returning 0 lets the caller try as-is.
        return 0
    return best_i


def make_unique_columns(cols: List[Any]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for c in cols:
        name = "" if pd.isna(c) else str(c).strip()
        if name == "":
            name = "Unnamed"
        base = name
        if base in seen:
            seen[base] += 1
            name = f"{base}.{seen[base]}"
        else:
            seen[base] = 0
        out.append(name)
    return out


def standardize_table_after_header(raw: pd.DataFrame, header_row: int) -> pd.DataFrame:
    header = make_unique_columns(raw.iloc[header_row].tolist())
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = header
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def read_uploaded_file(uploaded_file, sheet_name: Optional[str] = None) -> Tuple[pd.DataFrame, str, List[str]]:
    """Read uploaded Excel/CSV and return the worksheet-like table area."""
    name = uploaded_file.name
    ext = os.path.splitext(name)[1].lower()
    notes: List[str] = []

    data = uploaded_file.getvalue()

    if ext == ".csv":
        # CSV files usually already have headers.
        df = pd.read_csv(io.BytesIO(data))
        notes.append("Loaded CSV using the first row as headers.")
        return df.dropna(how="all").reset_index(drop=True), name, notes

    if ext in (".xlsx", ".xlsm", ".xls"):
        bio = io.BytesIO(data)
        xls = pd.ExcelFile(bio)
        selected = choose_sheet(xls.sheet_names, sheet_name)
        raw = pd.read_excel(io.BytesIO(data), sheet_name=selected, header=None)
        header_row = detect_header_row(raw)
        df = standardize_table_after_header(raw, header_row)
        notes.append(f"Loaded sheet '{selected}' and detected header row {header_row + 1}.")
        return df, selected, notes

    if ext == ".xlsb":
        # pyxlsb support is provided through pandas when pyxlsb is installed.
        bio = io.BytesIO(data)
        xls = pd.ExcelFile(bio, engine="pyxlsb")
        selected = choose_sheet(xls.sheet_names, sheet_name)
        raw = pd.read_excel(io.BytesIO(data), sheet_name=selected, header=None, engine="pyxlsb")
        header_row = detect_header_row(raw)
        df = standardize_table_after_header(raw, header_row)
        notes.append(f"Loaded sheet '{selected}' and detected header row {header_row + 1}.")
        return df, selected, notes

    raise ValueError("Unsupported file type. Upload .csv, .xlsx, .xlsm, .xls, or .xlsb.")


def choose_sheet(sheet_names: List[str], requested: Optional[str] = None) -> str:
    if requested and requested in sheet_names:
        return requested
    for hint in WORKING_TABLE_HINTS:
        for s in sheet_names:
            if s.strip().lower() == hint.strip().lower():
                return s
    for s in sheet_names:
        if "working" in s.lower() and "table" in s.lower():
            return s
    return sheet_names[0]


def find_target_column(raw_df: pd.DataFrame) -> Optional[str]:
    for c in raw_df.columns:
        if canonical_label(c) == TARGET_COL:
            return c
    return None


def find_flag_column(raw_df: pd.DataFrame) -> Optional[str]:
    for c in raw_df.columns:
        if canonical_label(c) == "Flag":
            return c
    return None


def flag_is_model_eligible(v: Any) -> bool:
    s = str(v).strip().upper()
    return ("ALLOC" in s and "NO" not in s) or ("REVIEW" in s)


def drop_unnecessary_rows(raw_df: pd.DataFrame, drop_non_model_rows: bool) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Drop blank/header/irrelevant rows while preserving table columns."""
    notes: List[str] = []
    df = raw_df.dropna(how="all").copy()
    before = len(df)

    # Remove accidental repeated header rows inside exported sheets.
    row_text = df.astype(str).agg("|".join, axis=1).str.upper()
    repeated_header_mask = row_text.str.contains("FINAL ALLOC", na=False) & row_text.str.contains("ALLOC", na=False)
    if repeated_header_mask.any():
        df = df.loc[~repeated_header_mask].copy()
        notes.append(f"Dropped {int(repeated_header_mask.sum())} repeated header row(s).")

    flag_col = find_flag_column(df)
    if flag_col is not None:
        flag_nonblank = df[flag_col].notna() & (df[flag_col].astype(str).str.strip() != "")
        if drop_non_model_rows:
            eligible = df[flag_col].map(flag_is_model_eligible).fillna(False).astype(bool)
            df = df.loc[eligible].copy()
            notes.append(f"Dropped {before - len(df):,} row(s) that were blank, repeated headers, or not Allocate/Review.")
        else:
            df = df.loc[flag_nonblank].copy()
            notes.append(f"Dropped {before - len(df):,} blank/no-flag row(s); kept all flagged rows.")
    else:
        notes.append("Could not find a Flag column, so only fully blank rows were removed before validation.")

    return df.reset_index(drop=True), df.index.to_numpy(), notes


def prepare_raw_for_prediction(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Validate that all model columns exist without leaking non-approved columns into model inference."""
    # canonicalize_columns performs strict allowed-column selection for the model.
    _ = canonicalize_columns(raw_df, target_required=False)
    return raw_df


def fill_final_alloc(raw_df: pd.DataFrame, pred_df: pd.DataFrame, blank_zero: bool = True) -> pd.DataFrame:
    out = raw_df.copy()
    target_col = find_target_column(out)
    if target_col is None:
        target_col = TARGET_COL
        out[target_col] = ""

    values = pred_df["Predicted Final Alloc"].tolist()
    clean_values = []
    for v in values:
        if v == "" or pd.isna(v):
            clean_values.append("")
        else:
            try:
                x = float(v)
                if x <= 0 and blank_zero:
                    clean_values.append("")
                elif abs(x - round(x)) < 1e-9:
                    clean_values.append(int(round(x)))
                else:
                    clean_values.append(round(x, 4))
            except Exception:
                clean_values.append(v)
    out[target_col] = clean_values
    return out


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


@st.cache_resource(show_spinner=False)
def get_model_bundle():
    return load_artifacts(MODEL_DIR)


@st.cache_data(show_spinner=False)
def cached_training_summary():
    summaries = {}
    for name in ["allocate_training_history.json", "review_training_history.json"]:
        path = os.path.join(APP_DIR, name)
        if os.path.exists(path):
            try:
                import json
                with open(path, "r") as f:
                    summaries[name] = json.load(f)
            except Exception:
                summaries[name] = None
    return summaries


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

st.title("📦 Allocation Split Expert Streamlit App")
st.caption("Two-stage Allocate/Review model using NumPy-only inference with DC-aware Review ranking.")

with st.sidebar:
    st.header("Model Controls")
    uploaded = st.file_uploader("Upload allocation sheet", type=["xlsx", "xlsm", "xls", "xlsb", "csv"])
    drop_non_model_rows = st.checkbox(
        "Drop rows that are not Allocate/Review",
        value=True,
        help="Recommended. The app removes rows that the split model is not designed to fill.",
    )
    blank_zero = st.checkbox("Return zero allocations as blank", value=True)
    st.divider()
    st.write("**Required model columns**")
    st.code("\n".join(ALLOWED_FEATURES), language="text")

bundle = get_model_bundle()
meta = bundle["meta"]
train_cfg = meta.get("train_config", {})

m1, m2, m3, m4 = st.columns(4)
m1.metric("Allocate threshold", f"{train_cfg.get('allocate_threshold', 0.42):.2f}")
m2.metric("Review threshold", f"{train_cfg.get('review_threshold', 0.58):.2f}")
m3.metric("Feature count", len(meta.get("feature_config", {}).get("feature_names", [])))
m4.metric("Model target", meta.get("target_col", TARGET_COL))

st.info(
    "This app uses only the approved columns for model features. Other columns are preserved in the exported CSV, "
    "but they are not used by the model."
)

if uploaded is None:
    st.subheader("How to use")
    st.markdown(
        """
        1. Upload the daily allocation workbook or CSV.  
        2. The app detects the working table/header row.  
        3. It drops unnecessary rows, builds the same engineered features used in training, and routes rows through the Allocate or Review model.  
        4. Review rows are ranked by allocation need and filled until the available DC pool is exhausted.  
        5. Download the CSV with `Final Alloc.` filled out.
        """
    )
    st.stop()

try:
    raw_df, sheet_used, load_notes = read_uploaded_file(uploaded)
    cleaned_df, kept_index, drop_notes = drop_unnecessary_rows(raw_df, drop_non_model_rows=drop_non_model_rows)
    prepare_raw_for_prediction(cleaned_df)

    with st.spinner("Running feature engineering and model allocation..."):
        pred_audit = predict_dataframe(cleaned_df, bundle, target_required=False)
        final_df = fill_final_alloc(cleaned_df, pred_audit, blank_zero=blank_zero)

    st.success("Allocation completed. Download the filled CSV below.")

    for note in load_notes + drop_notes:
        st.write(f"• {note}")

    # Summary metrics
    target_col = find_target_column(final_df) or TARGET_COL
    pred_numeric = pd.to_numeric(final_df[target_col], errors="coerce").fillna(0)
    flag_col = find_flag_column(final_df)
    flags = final_df[flag_col].astype(str).str.upper() if flag_col else pd.Series([""] * len(final_df))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows exported", f"{len(final_df):,}")
    c2.metric("Rows allocated", f"{int((pred_numeric > 0).sum()):,}")
    c3.metric("Total allocated units", f"{int(pred_numeric.sum()):,}")
    c4.metric("Review rows", f"{int(flags.str.contains('REVIEW', na=False).sum()):,}")

    output_name = os.path.splitext(uploaded.name)[0] + "_final_alloc_filled.csv"
    st.download_button(
        "Download filled CSV",
        data=to_csv_bytes(final_df),
        file_name=output_name,
        mime="text/csv",
        type="primary",
    )

    st.subheader("Preview: Final CSV")
    st.dataframe(final_df.head(250), use_container_width=True)

    st.subheader("Model Audit Preview")
    audit_cols = [c for c in ["Class Name", "Line Name", "Site", "Flag", "FLM", "Dc Avail", "Alloc. Rec.", "Predicted Final Alloc", "Predicted Group", "Allocation Confidence", "Raw Predicted FLMs"] if c in pred_audit.columns]
    st.dataframe(pred_audit[audit_cols].head(250), use_container_width=True)

    with st.expander("Column validation details"):
        canon = canonicalize_columns(cleaned_df, target_required=False)
        st.write("Canonical feature columns used by the model:")
        st.dataframe(canon.head(20), use_container_width=True)

except Exception as e:
    st.error("Prediction failed.")
    st.exception(e)
    st.markdown(
        """
        Common causes:
        - The uploaded file does not contain the required approved columns.
        - The working table header row was not detected.
        - The workbook sheet is not the allocation working table.
        - `.xlsb` support is missing from the environment; make sure `pyxlsb` is installed.
        """
    )
