from __future__ import annotations

import io
import os
import json
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
REPORT_ZIP = os.path.join(APP_DIR, "model_report_package.zip")

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
    """Detect worksheet header row using approved feature/target names."""
    wanted = set(ALLOWED_FEATURES + [TARGET_COL])
    best_i, best_hits = 0, -1
    search_rows = min(len(raw), 70)
    for i in range(search_rows):
        vals = raw.iloc[i].tolist()
        hits = sum(1 for v in vals if canonical_label(v) in wanted)
        if hits > best_hits:
            best_i, best_hits = i, hits
    if best_hits < min_hits:
        return 0
    return best_i


def make_unique_columns(cols: List[Any]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for c in cols:
        try:
            name = "" if pd.isna(c) else str(c).strip()
        except Exception:
            name = str(c).strip()
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


def read_uploaded_file(uploaded_file, sheet_name: Optional[str] = None) -> Tuple[pd.DataFrame, str, List[str]]:
    """Read uploaded Excel/CSV and return worksheet-like table area."""
    name = uploaded_file.name
    ext = os.path.splitext(name)[1].lower()
    notes: List[str] = []
    data = uploaded_file.getvalue()

    if ext == ".csv":
        df = pd.read_csv(io.BytesIO(data))
        notes.append("Loaded CSV using the first row as headers.")
        return df.dropna(how="all").reset_index(drop=True), name, notes

    if ext in (".xlsx", ".xlsm", ".xls"):
        xls = pd.ExcelFile(io.BytesIO(data))
        selected = choose_sheet(xls.sheet_names, sheet_name)
        raw = pd.read_excel(io.BytesIO(data), sheet_name=selected, header=None)
        header_row = detect_header_row(raw)
        df = standardize_table_after_header(raw, header_row)
        notes.append(f"Loaded sheet '{selected}' and detected header row {header_row + 1}.")
        return df, selected, notes

    if ext == ".xlsb":
        xls = pd.ExcelFile(io.BytesIO(data), engine="pyxlsb")
        selected = choose_sheet(xls.sheet_names, sheet_name)
        raw = pd.read_excel(io.BytesIO(data), sheet_name=selected, header=None, engine="pyxlsb")
        header_row = detect_header_row(raw)
        df = standardize_table_after_header(raw, header_row)
        notes.append(f"Loaded sheet '{selected}' and detected header row {header_row + 1}.")
        return df, selected, notes

    raise ValueError("Unsupported file type. Upload .csv, .xlsx, .xlsm, .xls, or .xlsb.")


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


def safe_cell_to_str(x: Any) -> str:
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def drop_unnecessary_rows(raw_df: pd.DataFrame, drop_non_model_rows: bool) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Drop blank/header/irrelevant rows while preserving table columns."""
    notes: List[str] = []
    df = raw_df.dropna(how="all").copy()
    before = len(df)

    row_text = df.apply(lambda row: "|".join(safe_cell_to_str(v) for v in row.to_numpy()), axis=1).str.upper()
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
    _ = canonicalize_columns(raw_df, target_required=False)
    return raw_df


def clean_alloc_value(v: Any, blank_zero: bool = True) -> Any:
    if v == "" or pd.isna(v):
        return ""
    try:
        x = float(v)
        if x <= 0 and blank_zero:
            return ""
        if abs(x - round(x)) < 1e-9:
            return int(round(x))
        return round(x, 4)
    except Exception:
        return v


def fill_final_alloc(raw_df: pd.DataFrame, pred_df: pd.DataFrame, blank_zero: bool = True) -> pd.DataFrame:
    out = raw_df.copy()
    target_col = find_target_column(out)
    if target_col is None:
        target_col = TARGET_COL
        out[target_col] = ""
    out[target_col] = [clean_alloc_value(v, blank_zero) for v in pred_df["Predicted Final Alloc"].tolist()]
    return out


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def bytes_from_file(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


@st.cache_resource(show_spinner=False)
def get_model_bundle():
    return load_artifacts(MODEL_DIR)


@st.cache_data(show_spinner=False)
def load_json_file(name: str):
    path = os.path.join(APP_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_feature_importance():
    path = os.path.join(APP_DIR, "model_feature_importance.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_detailed_feature_importance():
    path = os.path.join(APP_DIR, "model_feature_importance_detailed.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def summarize_history(hist: dict, key: str) -> dict:
    rows = hist.get(key, []) if isinstance(hist, dict) else []
    if not rows:
        return {}
    out = {"epochs_run": len(rows)}
    if key == "classifier":
        val_losses = [r for r in rows if r.get("val_loss") is not None]
        if val_losses:
            best = min(val_losses, key=lambda r: r["val_loss"])
            out["best_val_loss"] = best.get("val_loss")
            out["best_epoch"] = best.get("epoch")
        accs = [r.get("val_acc") for r in rows if r.get("val_acc") is not None]
        if accs:
            out["best_val_acc"] = max(accs)
    else:
        maes = [r for r in rows if r.get("val_mae_packs") is not None]
        if maes:
            best = min(maes, key=lambda r: r["val_mae_packs"])
            out["best_val_mae_packs"] = best.get("val_mae_packs")
            out["best_epoch"] = best.get("epoch")
    return out


def make_audit_table(cleaned_df: pd.DataFrame, pred_audit: pd.DataFrame, final_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_audit.copy()
    original_target_col = find_target_column(cleaned_df)
    if original_target_col is not None:
        original = cleaned_df[original_target_col]
    else:
        original = pd.Series([""] * len(cleaned_df))
    out["Original Final Alloc"] = original.values
    out["Output Final Alloc"] = final_df[find_target_column(final_df) or TARGET_COL].values
    old_num = pd.to_numeric(out["Original Final Alloc"], errors="coerce").fillna(0)
    new_num = pd.to_numeric(out["Output Final Alloc"], errors="coerce").fillna(0)
    out["Allocation Change"] = new_num - old_num
    out["Changed? "] = np.where(np.abs(out["Allocation Change"]) > 1e-9, "Changed", "Unchanged")
    out["Nonzero Prediction? "] = np.where(new_num > 0, "Nonzero", "Blank/Zero")
    return out


def filter_audit(audit: pd.DataFrame, flag_filter: str, result_filter: str, min_conf: float, max_conf: float) -> pd.DataFrame:
    view = audit.copy()
    if "Flag" in view.columns:
        f = view["Flag"].astype(str).str.upper()
        if flag_filter == "Allocate only":
            view = view.loc[f.str.contains("ALLOC", na=False) & ~f.str.contains("NO", na=False)]
        elif flag_filter == "Review only":
            view = view.loc[f.str.contains("REVIEW", na=False)]
    if result_filter == "Nonzero predictions only":
        view = view.loc[pd.to_numeric(view["Output Final Alloc"], errors="coerce").fillna(0) > 0]
    elif result_filter == "Blank/zero predictions only":
        view = view.loc[pd.to_numeric(view["Output Final Alloc"], errors="coerce").fillna(0) <= 0]
    elif result_filter == "Changed rows only":
        view = view.loc[view["Changed? "] == "Changed"]
    elif result_filter == "Model cut original allocation":
        view = view.loc[pd.to_numeric(view["Allocation Change"], errors="coerce").fillna(0) < 0]
    elif result_filter == "Model added allocation":
        view = view.loc[pd.to_numeric(view["Allocation Change"], errors="coerce").fillna(0) > 0]
    conf = pd.to_numeric(view["Allocation Confidence"], errors="coerce").fillna(0)
    view = view.loc[(conf >= min_conf) & (conf <= max_conf)]
    return view


def show_model_overview(bundle: dict):
    meta = bundle["meta"]
    train_cfg = meta.get("train_config", {})
    summary = load_json_file("model_summary.json") or {}
    alloc_hist = load_json_file("allocate_training_history.json") or {}
    review_hist = load_json_file("review_training_history.json") or {}

    st.subheader("What this model does")
    st.markdown(
        """
        This app uses a **split two-stage allocation system**. It does not use one generic model for every row.

        **Allocate rows** are passed through an Allocate classifier and an Allocate FLM regressor.  
        **Review rows** are passed through a separate Review classifier and Review FLM regressor, then ranked by allocation need so the highest-need Review rows receive inventory first.  
        The final step applies FLM rounding, caps allocations by `Dc Avail`, and allows a below-FLM allocation only when the remaining DC quantity is less than one FLM.
        """
    )

    a, b, c, d = st.columns(4)
    a.metric("Approved input columns", len(meta.get("allowed_features", [])))
    b.metric("Engineered feature count", len(meta.get("feature_config", {}).get("feature_names", [])))
    c.metric("Allocate threshold", f"{train_cfg.get('allocate_threshold', 0):.2f}")
    d.metric("Review threshold", f"{train_cfg.get('review_threshold', 0):.2f}")

    st.markdown("### How it was created")
    st.markdown(
        f"""
        - Training target: **`{meta.get('target_col', TARGET_COL)}`**.
        - Model inputs were restricted to the approved feature list only.
        - Categorical fields such as `Class Name`, `Line Name`, `Site`, `Rank`, `Flag`, and `Dc Avail Bucket` were converted into hashed NumPy feature vectors.
        - Numeric columns were combined with allocation-specific engineered features such as demand gaps, supply-to-demand ratios, projected-demand gaps, recommendation FLMs, DC FLMs, rank score, and demand flags.
        - Classifier hidden layers: **{train_cfg.get('hidden_classifier')}**.
        - Regressor hidden layers: **{train_cfg.get('hidden_regressor')}**.
        - Review ranking weights: **{train_cfg.get('review_rank_weight_probability')} probability / {train_cfg.get('review_rank_weight_demand')} demand pressure**.
        """
    )

    st.markdown("### Training output")
    rows = []
    for segment, hist in [("Allocate", alloc_hist), ("Review", review_hist)]:
        clf = summarize_history(hist, "classifier")
        reg = summarize_history(hist, "regressor")
        rows.append({
            "Segment": segment,
            "Classifier epochs": clf.get("epochs_run"),
            "Best classifier val loss": clf.get("best_val_loss"),
            "Best classifier val acc": clf.get("best_val_acc"),
            "Regressor epochs": reg.get("epochs_run"),
            "Best regressor MAE packs": reg.get("best_val_mae_packs"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("### Approved model columns")
    st.code("\n".join(meta.get("allowed_features", ALLOWED_FEATURES)), language="text")

    if os.path.exists(REPORT_ZIP):
        st.download_button(
            "Download model report package",
            data=bytes_from_file(REPORT_ZIP),
            file_name="model_report_package.zip",
            mime="application/zip",
        )


def show_feature_importance_tab():
    st.subheader("Feature usage report")
    st.caption("This report estimates feature usage from the first-layer absolute model weights, aggregated across model components. It is a practical model-inspection measure, not a causal explanation.")
    imp = load_feature_importance()
    detailed = load_detailed_feature_importance()
    if imp.empty:
        st.warning("Feature-importance package was not found.")
        return
    components = ["OVERALL_AVERAGE"] + sorted([x for x in imp["model_component"].unique().tolist() if x != "OVERALL_AVERAGE"])
    selected = st.selectbox("Model component", components)
    top_n = st.slider("Top feature groups", 5, 50, 25)
    view = imp.loc[imp["model_component"] == selected].sort_values("importance_pct", ascending=False).head(top_n)
    st.dataframe(view, use_container_width=True)
    st.bar_chart(view.set_index("feature_group")["importance_pct"])

    with st.expander("Detailed hashed/numeric feature weights"):
        if detailed.empty:
            st.write("Detailed report not available.")
        else:
            detail_component = st.selectbox("Detailed component", sorted(detailed["model_component"].unique().tolist()), key="detail_component")
            st.dataframe(
                detailed.loc[detailed["model_component"] == detail_component].sort_values("importance_pct", ascending=False).head(100),
                use_container_width=True,
            )


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

st.title("📦 Allocation Split Expert Streamlit App")
st.caption("Two-stage Allocate/Review NumPy model with feature report, audit filters, and DC-aware Review ranking.")

bundle = get_model_bundle()
meta = bundle["meta"]
train_cfg = meta.get("train_config", {})

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
    st.write("**Current model**")
    st.write(f"Created: `{meta.get('created_at', 'unknown')}`")
    st.write(f"Allocate threshold: `{train_cfg.get('allocate_threshold', 0.40)}`")
    st.write(f"Review threshold: `{train_cfg.get('review_threshold', 0.62)}`")
    if os.path.exists(REPORT_ZIP):
        st.download_button(
            "Download report package",
            data=bytes_from_file(REPORT_ZIP),
            file_name="model_report_package.zip",
            mime="application/zip",
            key="sidebar_report_download",
        )

m1, m2, m3, m4 = st.columns(4)
m1.metric("Allocate threshold", f"{train_cfg.get('allocate_threshold', 0.40):.2f}")
m2.metric("Review threshold", f"{train_cfg.get('review_threshold', 0.62):.2f}")
m3.metric("Engineered features", len(meta.get("feature_config", {}).get("feature_names", [])))
m4.metric("Target", meta.get("target_col", TARGET_COL))

if uploaded is None:
    tab1, tab2, tab3 = st.tabs(["Model Overview", "Feature Report", "How to Use"])
    with tab1:
        show_model_overview(bundle)
    with tab2:
        show_feature_importance_tab()
    with tab3:
        st.markdown(
            """
            1. Upload a daily allocation workbook or CSV.  
            2. The app detects the working-table header row and removes unnecessary rows.  
            3. It rebuilds the training feature engineering and routes rows through the Allocate or Review model.  
            4. Review rows are sorted by allocation need and filled until the available DC pool is exhausted.  
            5. Use the Spot Check tab to filter Review rows, Allocate rows, changed rows, nonzero predictions, and confidence bands.  
            6. Download the filled CSV and optional audit CSV.
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
        audit_table = make_audit_table(cleaned_df, pred_audit, final_df)

    target_col = find_target_column(final_df) or TARGET_COL
    pred_numeric = pd.to_numeric(final_df[target_col], errors="coerce").fillna(0)
    flag_col = find_flag_column(final_df)
    flags = final_df[flag_col].astype(str).str.upper() if flag_col else pd.Series([""] * len(final_df))

    st.success("Allocation completed. Use the tabs below to review, spot-check, and download output.")

    tab_output, tab_spot, tab_model, tab_features, tab_validation = st.tabs([
        "Output & Downloads",
        "Spot Check Output",
        "Model Overview",
        "Feature Report",
        "Validation Details",
    ])

    with tab_output:
        for note in load_notes + drop_notes:
            st.write(f"• {note}")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rows exported", f"{len(final_df):,}")
        c2.metric("Rows allocated", f"{int((pred_numeric > 0).sum()):,}")
        c3.metric("Total allocated units", f"{int(pred_numeric.sum()):,}")
        c4.metric("Allocate rows", f"{int((flags.str.contains('ALLOC', na=False) & ~flags.str.contains('NO', na=False)).sum()):,}")
        c5.metric("Review rows", f"{int(flags.str.contains('REVIEW', na=False).sum()):,}")

        output_name = os.path.splitext(uploaded.name)[0] + "_final_alloc_filled.csv"
        audit_name = os.path.splitext(uploaded.name)[0] + "_model_audit.csv"
        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button(
                "Download filled CSV",
                data=to_csv_bytes(final_df),
                file_name=output_name,
                mime="text/csv",
                type="primary",
            )
        with d2:
            st.download_button(
                "Download audit CSV",
                data=to_csv_bytes(audit_table),
                file_name=audit_name,
                mime="text/csv",
            )
        with d3:
            if os.path.exists(REPORT_ZIP):
                st.download_button(
                    "Download model report package",
                    data=bytes_from_file(REPORT_ZIP),
                    file_name="model_report_package.zip",
                    mime="application/zip",
                )

        st.subheader("Preview: Filled CSV")
        st.dataframe(final_df.head(300), use_container_width=True)

    with tab_spot:
        st.subheader("Spot-check model output")
        f1, f2, f3 = st.columns([1, 1, 2])
        with f1:
            flag_filter = st.selectbox("Flag filter", ["All", "Allocate only", "Review only"])
        with f2:
            result_filter = st.selectbox(
                "Result filter",
                [
                    "All",
                    "Nonzero predictions only",
                    "Blank/zero predictions only",
                    "Changed rows only",
                    "Model cut original allocation",
                    "Model added allocation",
                ],
            )
        with f3:
            min_conf, max_conf = st.slider("Allocation confidence range", 0.0, 1.0, (0.0, 1.0), 0.01)

        view = filter_audit(audit_table, flag_filter, result_filter, min_conf, max_conf)
        st.write(f"Showing **{len(view):,}** row(s).")
        default_cols = [
            "Class Name", "Line Name", "Site", "Flag", "MIL", "FLM", "Cost", "L30", "D30", "D60", "LW", "TTM",
            "Supply", "Dc Avail", "Rank", "Proj. Demand", "Alloc. Rec.", "Original Final Alloc", "Output Final Alloc",
            "Allocation Change", "Predicted Group", "Allocation Confidence", "Raw Predicted FLMs",
        ]
        cols = [c for c in default_cols if c in view.columns]
        st.dataframe(view[cols].sort_values(["Allocation Confidence", "Raw Predicted FLMs"], ascending=False).head(1000), use_container_width=True)

        st.markdown("### High-priority Review rows")
        review_view = filter_audit(audit_table, "Review only", "All", 0.0, 1.0)
        review_view = review_view.sort_values(["Allocation Confidence", "Raw Predicted FLMs"], ascending=False)
        st.dataframe(review_view[cols].head(250), use_container_width=True)

    with tab_model:
        show_model_overview(bundle)

    with tab_features:
        show_feature_importance_tab()

    with tab_validation:
        st.subheader("Column validation details")
        canon = canonicalize_columns(cleaned_df, target_required=False)
        st.write("Canonical feature columns used by the model:")
        st.dataframe(canon.head(50), use_container_width=True)
        st.write("Detected sheet / file:", sheet_used)
        st.write("Rows after cleanup:", len(cleaned_df))
        st.write("Columns preserved in exported CSV:", len(final_df.columns))

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
