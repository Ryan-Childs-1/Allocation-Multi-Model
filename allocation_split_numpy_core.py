"""
Allocation Split Expert - NumPy-only model core

This module intentionally keeps model math in NumPy only so the saved artifacts can be
loaded inside a lightweight Streamlit app later. Pandas is used only for file loading and
DataFrame handling in the trainer/app.

Allowed feature columns only:
    Class Name, Line Name, Site, MIL, FLM, Cost, L30, D30, D60, LW, TTM,
    Supply, Dc Avail, Rank, Proj. Demand, Alloc. Rec., Flag
Training target:
    Final Alloc.  (target only; never used as a feature)
"""

from __future__ import annotations

import os, re, json, math, time, hashlib, zipfile
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

# -----------------------------------------------------------------------------
# Column contract
# -----------------------------------------------------------------------------

ALLOWED_FEATURES = [
    "Class Name", "Line Name", "Site", "MIL", "FLM", "Cost", "L30", "D30", "D60",
    "LW", "TTM", "Supply", "Dc Avail", "Rank", "Proj. Demand", "Alloc. Rec.", "Flag"
]
TARGET_COL = "Final Alloc."

NUMERIC_COLS = [
    "MIL", "FLM", "Cost", "L30", "D30", "D60", "LW", "TTM", "Supply",
    "Dc Avail", "Proj. Demand", "Alloc. Rec."
]
CATEGORICAL_COLS = ["Class Name", "Line Name", "Site", "Rank", "Flag", "Dc Avail Bucket"]

# Project APE-style DC availability buckets expressed in FLMs.
# The label is generated from dc_avail / max(FLM, 1).
DC_AVAIL_BUCKETS = [
    ("DC_EMPTY", -np.inf, 0.0),
    ("DC_LT_1_FLM", 0.0, 1.0),
    ("DC_1_FLM", 1.0, 2.0),
    ("DC_2_3_FLM", 2.0, 4.0),
    ("DC_4_6_FLM", 4.0, 7.0),
    ("DC_7_12_FLM", 7.0, 13.0),
    ("DC_13_24_FLM", 13.0, 25.0),
    ("DC_25_PLUS_FLM", 25.0, np.inf),
]

GROUP_LABELS = [
    "NO_ALLOC",
    "ONE_FLM",
    "TWO_FLM",
    "THREE_TO_FOUR_FLM",
    "FIVE_TO_EIGHT_FLM",
    "NINE_PLUS_FLM",
]

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def _norm_name(x: Any) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"\.[0-9]+$", "", s)       # pandas duplicate suffixes: MIL.1 -> MIL
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

CANONICAL_ALIASES = {
    "class name": "Class Name",
    "line name": "Line Name",
    "site": "Site",
    "mil": "MIL",
    "flm": "FLM",
    "cost": "Cost",
    "l30": "L30",
    "d30": "D30",
    "d60": "D60",
    "lw": "LW",
    "ttm": "TTM",
    "supply": "Supply",
    "dc avail": "Dc Avail",
    "dc available": "Dc Avail",
    "rank": "Rank",
    "proj demand": "Proj. Demand",
    "projected demand": "Proj. Demand",
    "alloc rec": "Alloc. Rec.",
    "allocation rec": "Alloc. Rec.",
    "allocation recommendation": "Alloc. Rec.",
    "flag": "Flag",
    "final alloc": "Final Alloc.",
    "final allocation": "Final Alloc.",
}

def canonicalize_columns(df, target_required: bool = True):
    """Return a copy with only allowed features plus target.

    Handles duplicate worksheet columns such as MIL and FLM. If duplicates exist, this
    keeps the later occurrence for MIL/FLM because allocation worksheets commonly repeat
    these fields near Proj. Demand / Alloc. Rec.
    """
    import pandas as pd
    col_map: Dict[str, List[str]] = {}
    for c in df.columns:
        key = _norm_name(c)
        canon = CANONICAL_ALIASES.get(key)
        if canon:
            col_map.setdefault(canon, []).append(c)

    required = list(ALLOWED_FEATURES) + ([TARGET_COL] if target_required else [])
    missing = [c for c in required if c not in col_map]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found canonical columns: {sorted(col_map)}")

    out = pd.DataFrame(index=df.index)
    for canon in ALLOWED_FEATURES:
        candidates = col_map[canon]
        # Prefer later MIL/FLM duplicate because it appears in the allocation section.
        chosen = candidates[-1] if canon in ("MIL", "FLM") else candidates[0]
        out[canon] = df[chosen]
    if target_required:
        out[TARGET_COL] = df[col_map[TARGET_COL][0]]
    return out


def load_allocation_files(paths: List[str], sheet_name: str = "3.3 Working Table", target_required: bool = True):
    """Load csv/xlsx/xlsb files and return one canonical DataFrame.

    Pandas is used for file I/O only. Model math remains NumPy-only.
    """
    import pandas as pd
    frames = []
    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            raw = pd.read_csv(path)
        elif ext == ".xlsb":
            raw = pd.read_excel(path, sheet_name=sheet_name, engine="pyxlsb", header=1)
        elif ext in (".xlsx", ".xlsm", ".xls"):
            raw = pd.read_excel(path, sheet_name=sheet_name, header=1)
        else:
            raise ValueError(f"Unsupported file type: {path}")
        canon = canonicalize_columns(raw, target_required=target_required)
        canon["__source_file"] = os.path.basename(path)
        frames.append(canon)
    if not frames:
        raise ValueError("No input files supplied.")
    return pd.concat(frames, ignore_index=True)


# -----------------------------------------------------------------------------
# Pickle training dataset loader
# -----------------------------------------------------------------------------

def _find_first_key(d: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    norm_to_key = {_norm_name(k): k for k in d.keys()}
    for cand in candidates:
        key = norm_to_key.get(_norm_name(cand))
        if key is not None:
            return key
    # Also allow loose contains matching for names such as allocate_rows/review_rows.
    for k in d.keys():
        nk = _norm_name(k)
        for cand in candidates:
            nc = _norm_name(cand)
            if nc in nk or nk in nc:
                return k
    return None


def _coerce_split_object_to_df(obj: Any, segment_flag: str, parent_columns: Optional[List[str]] = None):
    """Coerce one separated pickle object into a canonical training DataFrame.

    Supported object shapes:
      1. pandas DataFrame with approved columns and Final Alloc.
      2. dict containing one DataFrame under df/data/dataframe/rows.
      3. dict containing X plus y/target/final_alloc. X can be a DataFrame or ndarray.
      4. tuple/list of (X, y) or (df, y).

    If the separated object has no Flag column, the loader injects Flag as either
    "Allocate" or "Review" based on the split being loaded.
    """
    import pandas as pd

    def attach_flag_and_target(raw_df, y=None):
        df = raw_df.copy()
        # Attach target if provided outside X.
        if y is not None and TARGET_COL not in [CANONICAL_ALIASES.get(_norm_name(c), c) for c in df.columns]:
            df[TARGET_COL] = np.asarray(y).reshape(-1)
        # Inject segment flag when separated pickle did not store Flag.
        has_flag = any(CANONICAL_ALIASES.get(_norm_name(c)) == "Flag" for c in df.columns)
        if not has_flag:
            df["Flag"] = segment_flag
        return canonicalize_columns(df, target_required=True)

    if isinstance(obj, pd.DataFrame):
        return attach_flag_and_target(obj)

    if isinstance(obj, (tuple, list)):
        if len(obj) == 0:
            raise ValueError(f"Empty {segment_flag} object in pickle.")
        if len(obj) >= 2:
            X, y = obj[0], obj[1]
            if isinstance(X, pd.DataFrame):
                return attach_flag_and_target(X, y)
            cols = parent_columns
            if cols is None and len(obj) >= 3 and isinstance(obj[2], (list, tuple)):
                cols = list(obj[2])
            if cols is None:
                raise ValueError(f"{segment_flag} tuple has ndarray X but no feature column names.")
            return attach_flag_and_target(pd.DataFrame(np.asarray(X), columns=list(cols)), y)
        if isinstance(obj[0], pd.DataFrame):
            return attach_flag_and_target(obj[0])

    if isinstance(obj, dict):
        columns_key = _find_first_key(obj, ["feature_columns", "columns", "feature_names", "allowed_features", "x_columns"])
        columns = list(obj[columns_key]) if columns_key is not None else parent_columns

        df_key = _find_first_key(obj, ["df", "dataframe", "data", "rows", "allocate_df", "review_df"])
        if df_key is not None and hasattr(obj[df_key], "columns"):
            y_key = _find_first_key(obj, ["y", "target", "targets", "Final Alloc.", "final_alloc", "final allocation"])
            y = obj[y_key] if y_key is not None else None
            return attach_flag_and_target(obj[df_key], y)

        x_key = _find_first_key(obj, ["X", "x", "features", "feature_matrix", "X_allocate", "X_review"])
        y_key = _find_first_key(obj, ["y", "target", "targets", "Final Alloc.", "final_alloc", "final allocation", "y_allocate", "y_review"])
        if x_key is not None:
            X = obj[x_key]
            y = obj[y_key] if y_key is not None else None
            if y is None:
                raise ValueError(f"{segment_flag} split has X/features but no target y/Final Alloc.")
            if isinstance(X, pd.DataFrame):
                return attach_flag_and_target(X, y)
            if columns is None:
                raise ValueError(f"{segment_flag} split has ndarray X but no feature column names.")
            return attach_flag_and_target(pd.DataFrame(np.asarray(X), columns=list(columns)), y)

    raise ValueError(f"Could not understand the {segment_flag} object in the pickle. Expected a DataFrame, dict, or (X, y) tuple.")


def load_two_model_pickle(path: str, target_required: bool = True):
    """Load `two_model_advanced_training_dataset.pkl` or similar separated datasets.

    The pickle is expected to contain Allocate and Review data already separated. The
    function combines them back into one canonical DataFrame for this trainer while
    preserving the Flag values required by the split model system.

    Supported top-level formats include:
      - {"allocate": allocate_df, "review": review_df}
      - {"allocate_df": ..., "review_df": ...}
      - {"X_allocate": ..., "y_allocate": ..., "X_review": ..., "y_review": ..., "feature_columns": [...]}
      - a single DataFrame containing both Flag values
      - a tuple/list containing two split objects: (allocate_obj, review_obj)
    """
    import pickle
    import pandas as pd

    with open(path, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, pd.DataFrame):
        canon = canonicalize_columns(payload, target_required=target_required)
        canon["__source_file"] = os.path.basename(path)
        return canon

    frames = []
    parent_columns = None
    if isinstance(payload, dict):
        col_key = _find_first_key(payload, ["feature_columns", "columns", "feature_names", "allowed_features", "x_columns"])
        if col_key is not None:
            parent_columns = list(payload[col_key])

        # Common nested split keys.
        alloc_key = _find_first_key(payload, ["allocate", "alloc", "allocate_df", "alloc_df", "allocate_rows", "alloc_rows", "train_allocate"])
        review_key = _find_first_key(payload, ["review", "review_df", "review_rows", "train_review"])

        # Common flat X/y key layout.
        if alloc_key is None and any(_norm_name(k) in ("x allocate", "y allocate") for k in payload.keys()):
            alloc_key = "__flat_allocate__"
        if review_key is None and any(_norm_name(k) in ("x review", "y review") for k in payload.keys()):
            review_key = "__flat_review__"

        if alloc_key == "__flat_allocate__":
            alloc_obj = {
                "X": payload[_find_first_key(payload, ["X_allocate", "x_allocate", "allocate_X", "alloc_X"])],
                "y": payload[_find_first_key(payload, ["y_allocate", "target_allocate", "allocate_y", "alloc_y"])],
                "feature_columns": parent_columns,
            }
        elif alloc_key is not None:
            alloc_obj = payload[alloc_key]
        else:
            alloc_obj = None

        if review_key == "__flat_review__":
            review_obj = {
                "X": payload[_find_first_key(payload, ["X_review", "x_review", "review_X"])],
                "y": payload[_find_first_key(payload, ["y_review", "target_review", "review_y"])],
                "feature_columns": parent_columns,
            }
        elif review_key is not None:
            review_obj = payload[review_key]
        else:
            review_obj = None

        if alloc_obj is not None:
            a = _coerce_split_object_to_df(alloc_obj, "Allocate", parent_columns=parent_columns)
            a["__split_source"] = "allocate"
            frames.append(a)
        if review_obj is not None:
            r = _coerce_split_object_to_df(review_obj, "Review", parent_columns=parent_columns)
            r["__split_source"] = "review"
            frames.append(r)

    elif isinstance(payload, (tuple, list)) and len(payload) >= 2:
        a = _coerce_split_object_to_df(payload[0], "Allocate")
        r = _coerce_split_object_to_df(payload[1], "Review")
        a["__split_source"] = "allocate"
        r["__split_source"] = "review"
        frames.extend([a, r])

    if not frames:
        raise ValueError(
            "Could not find separated Allocate and Review data in the pickle. "
            "Use keys like 'allocate'/'review', 'allocate_df'/'review_df', or "
            "X_allocate/y_allocate and X_review/y_review with feature_columns."
        )

    out = pd.concat(frames, ignore_index=True)
    out["__source_file"] = os.path.basename(path)
    return out


def load_training_dataset(paths: List[str], sheet_name: str = "3.3 Working Table", target_required: bool = True):
    """Load training data from either pickle datasets or workbook/CSV files.

    If any path is a .pkl/.pickle, this function loads those files with
    load_two_model_pickle. Other paths are loaded with load_allocation_files.
    """
    import pandas as pd
    pickle_paths = [p for p in paths if os.path.splitext(p)[1].lower() in (".pkl", ".pickle")]
    other_paths = [p for p in paths if p not in pickle_paths]
    frames = []
    for p in pickle_paths:
        frames.append(load_two_model_pickle(p, target_required=target_required))
    if other_paths:
        frames.append(load_allocation_files(other_paths, sheet_name=sheet_name, target_required=target_required))
    if not frames:
        raise ValueError("No input files supplied.")
    return pd.concat(frames, ignore_index=True)


def to_float_array(series, default: float = 0.0):
    import pandas as pd
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    return s.astype(float).to_numpy()


def clean_flag(values) -> np.ndarray:
    return np.array([str(v).strip().upper() if not (v is None or (isinstance(v, float) and np.isnan(v))) else "" for v in values])


def flag_mask(flags: np.ndarray, kind: str) -> np.ndarray:
    k = kind.upper()
    if k == "ALLOCATE":
        return np.array([("ALLOC" in f) and ("NO" not in f) for f in flags])
    if k == "REVIEW":
        return np.array(["REVIEW" in f for f in flags])
    return np.zeros(len(flags), dtype=bool)


def rank_to_score(x: Any) -> float:
    s = str(x).strip().upper()
    mapping = {"A+": 1.20, "A": 1.00, "B": 0.75, "C": 0.50, "D": 0.25, "E": 0.10, "F": 0.05}
    return mapping.get(s, 0.50)


def make_dc_bucket(dc_avail: np.ndarray, flm: np.ndarray) -> np.ndarray:
    ratio = dc_avail / np.maximum(flm, 1.0)
    labels = np.empty(len(ratio), dtype=object)
    for label, lo, hi in DC_AVAIL_BUCKETS:
        mask = (ratio > lo) & (ratio <= hi) if lo == -np.inf else (ratio >= lo) & (ratio < hi)
        labels[mask] = label
    return labels


def stable_hash(s: Any, mod: int) -> int:
    raw = str(s).strip().upper().encode("utf-8")
    return int(hashlib.md5(raw).hexdigest()[:8], 16) % mod

@dataclass
class FeatureConfig:
    hash_dim_class: int = 96
    hash_dim_line: int = 128
    hash_dim_site: int = 96
    hash_dim_rank: int = 8
    hash_dim_flag: int = 8
    hash_dim_dc_bucket: int = 8
    numeric_mean: Optional[List[float]] = None
    numeric_std: Optional[List[float]] = None
    feature_names: Optional[List[str]] = None


def build_features(df, config: Optional[FeatureConfig] = None, fit: bool = False) -> Tuple[np.ndarray, FeatureConfig]:
    """Create engineered features using ONLY allowed columns."""
    import pandas as pd
    if config is None:
        config = FeatureConfig()

    n = len(df)
    vals = {c: to_float_array(df[c], default=0.0) for c in NUMERIC_COLS}
    flm = np.maximum(vals["FLM"], 1.0)
    dc_avail = vals["Dc Avail"]
    supply = vals["Supply"]
    l30, d30, d60, lw, ttm = vals["L30"], vals["D30"], vals["D60"], vals["LW"], vals["TTM"]
    proj = vals["Proj. Demand"]
    rec = vals["Alloc. Rec."]
    mil = vals["MIL"]
    cost = vals["Cost"]

    dc_bucket = make_dc_bucket(dc_avail, flm)
    rank_score = np.array([rank_to_score(x) for x in df["Rank"].values], dtype=float)

    max_demand = np.maximum.reduce([d30, d60 / 2.0, proj, l30, lw * 4.29, ttm / 12.0])
    avg_demand = (l30 + d30 + d60 / 2.0 + lw * 4.29 + ttm / 12.0 + proj) / 6.0
    demand_pressure = max_demand - supply
    proj_gap = proj - supply
    d30_gap = d30 - supply
    d60_gap = d60 - supply
    mil_gap = mil - supply
    rec_flms = rec / flm
    dc_flms = dc_avail / flm
    weak_demand = ((l30 <= 0) & (d30 <= 0) & (d60 <= 0) & (lw <= 0)).astype(float)

    numeric_features = [
        # raw allowed numeric columns
        vals["MIL"], flm, cost, l30, d30, d60, lw, ttm, supply, dc_avail, proj, rec,
        # demand/supply engineering
        max_demand, avg_demand, demand_pressure, proj_gap, d30_gap, d60_gap, mil_gap,
        supply / np.maximum(d30, 1.0), supply / np.maximum(d60, 1.0), supply / np.maximum(proj, 1.0),
        l30 / np.maximum(d30, 1.0), d30 / np.maximum(d60, 1.0), (lw * 4.29) / np.maximum(l30, 1.0),
        ttm / 12.0, rec_flms, rec - np.maximum(demand_pressure, 0.0), dc_flms,
        np.minimum(dc_flms, 25.0), rank_score, cost * np.maximum(rec, 0.0), cost * np.maximum(supply, 0.0), weak_demand,
        (supply <= 0).astype(float), (rec > 0).astype(float), (dc_avail > 0).astype(float),
    ]
    names = [
        "MIL", "FLM", "Cost", "L30", "D30", "D60", "LW", "TTM", "Supply", "Dc Avail", "Proj. Demand", "Alloc. Rec.",
        "max_demand", "avg_demand", "demand_pressure", "proj_gap", "d30_gap", "d60_gap", "mil_gap",
        "supply_to_d30", "supply_to_d60", "supply_to_proj", "l30_to_d30", "d30_to_d60", "lw_month_to_l30",
        "ttm_monthly", "rec_flms", "rec_minus_pressure", "dc_flms", "dc_flms_capped", "rank_score",
        "rec_cost_value", "supply_cost_value", "weak_demand", "zero_supply", "has_alloc_rec", "has_dc_avail",
    ]
    X_num = np.vstack(numeric_features).T.astype(np.float32)

    if fit:
        mean = np.nanmean(X_num, axis=0)
        std = np.nanstd(X_num, axis=0)
        std[std < 1e-6] = 1.0
        config.numeric_mean = mean.tolist()
        config.numeric_std = std.tolist()
    mean = np.array(config.numeric_mean, dtype=np.float32)
    std = np.array(config.numeric_std, dtype=np.float32)
    X_num = np.nan_to_num((X_num - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)

    # hashed one-hot categorical features; no sklearn dependency and supports unseen values.
    cat_specs = [
        ("Class Name", config.hash_dim_class), ("Line Name", config.hash_dim_line), ("Site", config.hash_dim_site),
        ("Rank", config.hash_dim_rank), ("Flag", config.hash_dim_flag), ("Dc Avail Bucket", config.hash_dim_dc_bucket),
    ]
    cat_values = {
        "Class Name": df["Class Name"].fillna("__MISSING__").astype(str).values,
        "Line Name": df["Line Name"].fillna("__MISSING__").astype(str).values,
        "Site": df["Site"].fillna("__MISSING__").astype(str).values,
        "Rank": df["Rank"].fillna("__MISSING__").astype(str).values,
        "Flag": df["Flag"].fillna("__MISSING__").astype(str).values,
        "Dc Avail Bucket": dc_bucket,
    }
    cat_mats = []
    cat_names = []
    for c, dim in cat_specs:
        mat = np.zeros((n, dim), dtype=np.float32)
        for i, v in enumerate(cat_values[c]):
            mat[i, stable_hash(v, dim)] = 1.0
        cat_mats.append(mat)
        cat_names += [f"hash__{c}__{j}" for j in range(dim)]

    X = np.hstack([X_num] + cat_mats).astype(np.float32)
    if fit:
        config.feature_names = names + cat_names
    return X, config


def make_pack_targets(df) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    final_alloc = to_float_array(df[TARGET_COL], default=0.0)
    flm = np.maximum(to_float_array(df["FLM"], default=1.0), 1.0)
    packs = final_alloc / flm
    positive = final_alloc > 0
    groups = np.zeros(len(df), dtype=np.int64)
    groups[(packs > 0) & (packs <= 1.25)] = 1
    groups[(packs > 1.25) & (packs <= 2.25)] = 2
    groups[(packs > 2.25) & (packs <= 4.25)] = 3
    groups[(packs > 4.25) & (packs <= 8.25)] = 4
    groups[packs > 8.25] = 5
    return positive.astype(np.float32), packs.astype(np.float32), groups

# -----------------------------------------------------------------------------
# NumPy neural models
# -----------------------------------------------------------------------------

class NumpyMLP:
    def __init__(self, input_dim: int, output_dim: int, hidden=(192, 96), task="softmax", seed=42, dropout=0.0):
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden = tuple(int(h) for h in hidden)
        self.task = task
        self.dropout = float(dropout)
        self.rng = np.random.default_rng(seed)
        dims = [self.input_dim] + list(self.hidden) + [self.output_dim]
        self.W, self.b = [], []
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / max(dims[i], 1))
            self.W.append((self.rng.normal(0, scale, size=(dims[i], dims[i+1]))).astype(np.float32))
            self.b.append(np.zeros(dims[i+1], dtype=np.float32))
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def forward(self, X, train=False):
        A = X.astype(np.float32)
        acts = [A]
        masks = []
        for i in range(len(self.W) - 1):
            Z = A @ self.W[i] + self.b[i]
            A = np.maximum(Z, 0)
            if train and self.dropout > 0:
                mask = (self.rng.random(A.shape) > self.dropout).astype(np.float32) / (1.0 - self.dropout)
                A *= mask
            else:
                mask = None
            acts.append(A)
            masks.append(mask)
        out = A @ self.W[-1] + self.b[-1]
        acts.append(out)
        return out, acts, masks

    @staticmethod
    def softmax(z):
        z = z - np.max(z, axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)

    def predict_proba(self, X):
        logits, _, _ = self.forward(X, train=False)
        return self.softmax(logits)

    def predict(self, X):
        out, _, _ = self.forward(X, train=False)
        if self.task == "softmax":
            return np.argmax(self.softmax(out), axis=1)
        return out.reshape(-1)

    def _adam_step(self, grads_W, grads_b, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for i in range(len(self.W)):
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * grads_W[i]
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * (grads_W[i] ** 2)
            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * grads_b[i]
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * (grads_b[i] ** 2)
            mW_hat = self.mW[i] / (1 - beta1 ** self.t)
            vW_hat = self.vW[i] / (1 - beta2 ** self.t)
            mb_hat = self.mb[i] / (1 - beta1 ** self.t)
            vb_hat = self.vb[i] / (1 - beta2 ** self.t)
            self.W[i] -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            self.b[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

    def fit_classifier(self, X, y, X_val=None, y_val=None, epochs=80, batch_size=512, lr=1e-3, class_weights=None, patience=12, verbose=True):
        y = y.astype(np.int64)
        n = len(X)
        if class_weights is None:
            counts = np.bincount(y, minlength=self.output_dim).astype(float)
            class_weights = counts.sum() / np.maximum(counts, 1.0)
            class_weights = class_weights / np.mean(class_weights)
        class_weights = np.array(class_weights, dtype=np.float32)
        best_state, best_val, no_improve = self.state_dict(), float("inf"), 0
        hist = []
        for ep in range(1, epochs + 1):
            idx = self.rng.permutation(n)
            losses = []
            for start in range(0, n, batch_size):
                bi = idx[start:start+batch_size]
                xb, yb = X[bi], y[bi]
                logits, acts, masks = self.forward(xb, train=True)
                p = self.softmax(logits)
                w = class_weights[yb]
                loss = -np.mean(w * np.log(np.maximum(p[np.arange(len(yb)), yb], 1e-9)))
                losses.append(loss)
                dZ = p
                dZ[np.arange(len(yb)), yb] -= 1.0
                dZ *= (w / np.maximum(np.mean(w), 1e-6))[:, None]
                dZ /= len(yb)
                grads_W, grads_b = self._backprop(dZ, acts)
                self._adam_step(grads_W, grads_b, lr)
            tr_loss = float(np.mean(losses))
            val_loss = tr_loss
            val_acc = None
            if X_val is not None and len(X_val):
                pv = self.predict_proba(X_val)
                val_loss = float(-np.mean(np.log(np.maximum(pv[np.arange(len(y_val)), y_val.astype(int)], 1e-9))))
                val_acc = float(np.mean(np.argmax(pv, axis=1) == y_val))
                if val_loss < best_val:
                    best_val, best_state, no_improve = val_loss, self.state_dict(), 0
                else:
                    no_improve += 1
            hist.append({"epoch": ep, "train_loss": tr_loss, "val_loss": val_loss, "val_acc": val_acc})
            if verbose and (ep == 1 or ep % 5 == 0):
                print(f"classifier epoch {ep:03d} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f}" + (f" | val_acc={val_acc:.3f}" if val_acc is not None else ""))
            if X_val is not None and no_improve >= patience:
                if verbose: print(f"early stopping at epoch {ep}; best_val={best_val:.4f}")
                break
        if X_val is not None:
            self.load_state_dict(best_state)
        return hist

    def fit_regressor(self, X, y, sample_weight=None, X_val=None, y_val=None, epochs=100, batch_size=512, lr=8e-4, huber_delta=1.0, patience=15, verbose=True):
        y = y.astype(np.float32).reshape(-1, 1)
        n = len(X)
        if sample_weight is None:
            sample_weight = np.ones(n, dtype=np.float32)
        sample_weight = sample_weight.astype(np.float32).reshape(-1, 1)
        best_state, best_val, no_improve = self.state_dict(), float("inf"), 0
        hist = []
        for ep in range(1, epochs + 1):
            idx = self.rng.permutation(n)
            losses = []
            for start in range(0, n, batch_size):
                bi = idx[start:start+batch_size]
                xb, yb, wb = X[bi], y[bi], sample_weight[bi]
                pred, acts, masks = self.forward(xb, train=True)
                err = pred - yb
                abs_err = np.abs(err)
                quad = np.minimum(abs_err, huber_delta)
                lin = abs_err - quad
                loss = np.mean(wb * (0.5 * quad ** 2 + huber_delta * lin))
                losses.append(loss)
                grad = np.where(abs_err <= huber_delta, err, huber_delta * np.sign(err))
                grad *= wb / np.maximum(np.mean(wb), 1e-6)
                grad /= len(yb)
                grads_W, grads_b = self._backprop(grad, acts)
                self._adam_step(grads_W, grads_b, lr)
            tr_loss = float(np.mean(losses))
            val_loss = tr_loss
            if X_val is not None and len(X_val):
                predv = self.predict(X_val).reshape(-1)
                val_loss = float(np.mean(np.abs(predv - y_val.reshape(-1))))
                if val_loss < best_val:
                    best_val, best_state, no_improve = val_loss, self.state_dict(), 0
                else:
                    no_improve += 1
            hist.append({"epoch": ep, "train_loss": tr_loss, "val_mae_packs": val_loss})
            if verbose and (ep == 1 or ep % 5 == 0):
                print(f"regressor epoch {ep:03d} | train_huber={tr_loss:.4f} | val_mae_packs={val_loss:.4f}")
            if X_val is not None and no_improve >= patience:
                if verbose: print(f"early stopping at epoch {ep}; best_val_mae={best_val:.4f}")
                break
        if X_val is not None:
            self.load_state_dict(best_state)
        return hist

    def _backprop(self, dA, acts):
        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)
        d = dA.astype(np.float32)
        for i in reversed(range(len(self.W))):
            A_prev = acts[i]
            grads_W[i] = A_prev.T @ d
            grads_b[i] = d.sum(axis=0)
            if i > 0:
                d = d @ self.W[i].T
                d = d * (acts[i] > 0)
        return grads_W, grads_b

    def state_dict(self):
        return {"W": [w.copy() for w in self.W], "b": [b.copy() for b in self.b], "input_dim": self.input_dim, "output_dim": self.output_dim, "hidden": self.hidden, "task": self.task, "dropout": self.dropout}

    def load_state_dict(self, state):
        self.W = [w.copy() for w in state["W"]]
        self.b = [b.copy() for b in state["b"]]
        self.input_dim = int(state["input_dim"])
        self.output_dim = int(state["output_dim"])
        self.hidden = tuple(state["hidden"])
        self.task = state["task"]
        self.dropout = float(state.get("dropout", 0.0))
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def save_npz(self, path):
        data = {"meta": np.array(json.dumps({"input_dim": self.input_dim, "output_dim": self.output_dim, "hidden": self.hidden, "task": self.task, "dropout": self.dropout}), dtype=object)}
        for i, w in enumerate(self.W): data[f"W{i}"] = w
        for i, b in enumerate(self.b): data[f"b{i}"] = b
        np.savez_compressed(path, **data)

    @classmethod
    def load_npz(cls, path):
        z = np.load(path, allow_pickle=True)
        meta = json.loads(str(z["meta"].item()))
        model = cls(meta["input_dim"], meta["output_dim"], tuple(meta["hidden"]), meta["task"], dropout=meta.get("dropout", 0.0))
        model.W = [z[f"W{i}"].astype(np.float32) for i in range(len(meta["hidden"]) + 1)]
        model.b = [z[f"b{i}"].astype(np.float32) for i in range(len(meta["hidden"]) + 1)]
        model.mW = [np.zeros_like(w) for w in model.W]
        model.vW = [np.zeros_like(w) for w in model.W]
        model.mb = [np.zeros_like(b) for b in model.b]
        model.vb = [np.zeros_like(b) for b in model.b]
        return model

# -----------------------------------------------------------------------------
# Split model trainer and predictor
# -----------------------------------------------------------------------------

@dataclass
class TrainConfig:
    hidden_classifier: Tuple[int, int] = (192, 96)
    hidden_regressor: Tuple[int, int, int] = (256, 128, 64)
    epochs_classifier: int = 80
    epochs_regressor: int = 110
    batch_size: int = 512
    lr_classifier: float = 9e-4
    lr_regressor: float = 7e-4
    validation_frac: float = 0.18
    seed: int = 42
    allocate_threshold: float = 0.42
    review_threshold: float = 0.58
    review_rank_weight_demand: float = 0.35
    review_rank_weight_probability: float = 0.65


def train_val_split(n: int, frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    nv = max(1, int(n * frac)) if n > 20 else max(0, int(n * frac))
    return idx[nv:], idx[:nv]


def train_flag_model(name: str, X: np.ndarray, groups: np.ndarray, packs: np.ndarray, positive: np.ndarray, cfg: TrainConfig):
    print(f"\n=== Training {name.upper()} two-stage model ===")
    print(f"Rows: {len(X):,} | positive final allocations: {int(positive.sum()):,} ({positive.mean():.1%})")
    tr, va = train_val_split(len(X), cfg.validation_frac, cfg.seed + (17 if name == "review" else 3))
    clf = NumpyMLP(X.shape[1], len(GROUP_LABELS), hidden=cfg.hidden_classifier, task="softmax", seed=cfg.seed, dropout=0.04)
    counts = np.bincount(groups[tr], minlength=len(GROUP_LABELS)).astype(float)
    cw = counts.sum() / np.maximum(counts, 1.0)
    cw = np.clip(cw / np.mean(cw), 0.35, 6.0)
    clf_hist = clf.fit_classifier(X[tr], groups[tr], X[va], groups[va], epochs=cfg.epochs_classifier,
                                  batch_size=cfg.batch_size, lr=cfg.lr_classifier,
                                  class_weights=cw, verbose=True)
    pos_idx = np.where(positive > 0)[0]
    reg_hist = []
    reg = None
    if len(pos_idx) >= 20:
        trp, vap = train_val_split(len(pos_idx), cfg.validation_frac, cfg.seed + (29 if name == "review" else 11))
        pi_tr, pi_va = pos_idx[trp], pos_idx[vap]
        reg = NumpyMLP(X.shape[1], 1, hidden=cfg.hidden_regressor, task="regression", seed=cfg.seed + 100, dropout=0.03)
        # Higher weight to larger pack cases, but not enough to let them dominate.
        sw = 1.0 + np.minimum(packs[pi_tr], 10.0) / 10.0
        reg_hist = reg.fit_regressor(X[pi_tr], packs[pi_tr], sample_weight=sw, X_val=X[pi_va], y_val=packs[pi_va],
                                     epochs=cfg.epochs_regressor, batch_size=cfg.batch_size,
                                     lr=cfg.lr_regressor, huber_delta=1.0, verbose=True)
    else:
        print(f"Skipping {name} regressor: not enough positive rows.")
    return {"classifier": clf, "regressor": reg, "classifier_history": clf_hist, "regressor_history": reg_hist}


def train_all_models(df, cfg: TrainConfig = TrainConfig()):
    X, feat_cfg = build_features(df, config=None, fit=True)
    positive, packs, groups = make_pack_targets(df)
    flags = clean_flag(df["Flag"].values)
    alloc_mask = flag_mask(flags, "ALLOCATE")
    review_mask = flag_mask(flags, "REVIEW")
    print(f"Total rows: {len(df):,}")
    print(f"Allocate rows: {alloc_mask.sum():,} | Review rows: {review_mask.sum():,}")
    models = {}
    if alloc_mask.sum() > 20:
        models["allocate"] = train_flag_model("allocate", X[alloc_mask], groups[alloc_mask], packs[alloc_mask], positive[alloc_mask], cfg)
    else:
        raise ValueError("Not enough Allocate rows to train.")
    if review_mask.sum() > 20:
        models["review"] = train_flag_model("review", X[review_mask], groups[review_mask], packs[review_mask], positive[review_mask], cfg)
    else:
        raise ValueError("Not enough Review rows to train.")
    return {"feature_config": feat_cfg, "train_config": asdict(cfg), "models": models, "group_labels": GROUP_LABELS}


def save_artifacts(bundle: Dict[str, Any], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "allowed_features": ALLOWED_FEATURES,
        "target_col": TARGET_COL,
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "dc_avail_buckets": [(a, float(b) if np.isfinite(b) else str(b), float(c) if np.isfinite(c) else str(c)) for a,b,c in DC_AVAIL_BUCKETS],
        "group_labels": GROUP_LABELS,
        "feature_config": asdict(bundle["feature_config"]),
        "train_config": bundle["train_config"],
    }
    with open(os.path.join(out_dir, "model_config.json"), "w") as f:
        json.dump(meta, f, indent=2)
    for segment in ["allocate", "review"]:
        seg = bundle["models"][segment]
        seg["classifier"].save_npz(os.path.join(out_dir, f"{segment}_group_classifier.npz"))
        if seg["regressor"] is not None:
            seg["regressor"].save_npz(os.path.join(out_dir, f"{segment}_flm_regressor.npz"))
        with open(os.path.join(out_dir, f"{segment}_training_history.json"), "w") as f:
            json.dump({"classifier": seg["classifier_history"], "regressor": seg["regressor_history"]}, f, indent=2)
    return out_dir


def zip_artifacts(out_dir: str, zip_path: Optional[str] = None):
    if zip_path is None:
        zip_path = out_dir.rstrip(os.sep) + ".zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fn in os.listdir(out_dir):
            z.write(os.path.join(out_dir, fn), arcname=fn)
    return zip_path


def load_artifacts(model_dir: str):
    with open(os.path.join(model_dir, "model_config.json"), "r") as f:
        meta = json.load(f)
    feat_cfg = FeatureConfig(**meta["feature_config"])
    models = {
        "allocate": {
            "classifier": NumpyMLP.load_npz(os.path.join(model_dir, "allocate_group_classifier.npz")),
            "regressor": NumpyMLP.load_npz(os.path.join(model_dir, "allocate_flm_regressor.npz")),
        },
        "review": {
            "classifier": NumpyMLP.load_npz(os.path.join(model_dir, "review_group_classifier.npz")),
            "regressor": NumpyMLP.load_npz(os.path.join(model_dir, "review_flm_regressor.npz")),
        },
    }
    return {"meta": meta, "feature_config": feat_cfg, "models": models}


def row_need_score(df, proba_positive, raw_packs):
    vals = {c: to_float_array(df[c], default=0.0) for c in NUMERIC_COLS}
    flm = np.maximum(vals["FLM"], 1.0)
    supply = vals["Supply"]
    max_demand = np.maximum.reduce([vals["D30"], vals["D60"] / 2.0, vals["Proj. Demand"], vals["L30"], vals["LW"] * 4.29, vals["TTM"] / 12.0])
    pressure = np.maximum(max_demand - supply, 0.0) / flm
    pressure_norm = pressure / np.maximum(np.nanpercentile(pressure, 90), 1.0)
    rank = np.array([rank_to_score(x) for x in df["Rank"].values], dtype=float)
    return (0.65 * proba_positive + 0.35 * np.clip(pressure_norm, 0, 2)) * np.maximum(rank, 0.1) * np.maximum(raw_packs, 0.1)


def predict_dataframe(df, artifact_bundle, target_required: bool = False):
    """Predict Final Alloc for a canonical DataFrame or raw worksheet DataFrame.

    The output respects the two separate row types:
      - Allocate rows are predicted independently using the allocate model.
      - Review rows are ranked by need and allocated until Dc Avail is empty.

    Because only the allowed columns are available, the review optimizer treats rows with
    equal Class Name + Line Name + Cost + initial Dc Avail as a DC pool. If a future app can
    include Item/Product ID, replace this pool key with the true item/DC key.
    """
    import pandas as pd
    try:
        work = canonicalize_columns(df, target_required=target_required)
    except Exception:
        work = df.copy()
    X, _ = build_features(work, config=artifact_bundle["feature_config"], fit=False)
    flags = clean_flag(work["Flag"].values)
    alloc_mask = flag_mask(flags, "ALLOCATE")
    review_mask = flag_mask(flags, "REVIEW")
    n = len(work)
    pred = np.full(n, np.nan, dtype=float)
    group_pred = np.full(n, "", dtype=object)
    conf = np.zeros(n, dtype=float)
    raw_packs_all = np.zeros(n, dtype=float)

    for segment, mask, threshold in [("allocate", alloc_mask, artifact_bundle["meta"].get("train_config", {}).get("allocate_threshold", 0.42)),
                                     ("review", review_mask, artifact_bundle["meta"].get("train_config", {}).get("review_threshold", 0.58))]:
        idx = np.where(mask)[0]
        if len(idx) == 0: continue
        clf = artifact_bundle["models"][segment]["classifier"]
        reg = artifact_bundle["models"][segment]["regressor"]
        p = clf.predict_proba(X[idx])
        positive_prob = 1.0 - p[:, 0]
        g = np.argmax(p, axis=1)
        raw_packs = np.maximum(reg.predict(X[idx]), 0.0) if reg is not None else np.maximum(g.astype(float), 0.0)
        conf[idx] = positive_prob
        raw_packs_all[idx] = raw_packs
        group_pred[idx] = np.array(GROUP_LABELS, dtype=object)[g]
        if segment == "allocate":
            flm = np.maximum(to_float_array(work.iloc[idx]["FLM"], 1.0), 1.0)
            dc = to_float_array(work.iloc[idx]["Dc Avail"], 0.0)
            units = raw_packs * flm
            rounded = np.where((positive_prob >= threshold) & (dc > 0), np.round(units / flm) * flm, np.nan)
            rounded = np.minimum(rounded, dc)
            # If model wants allocation and remaining DC is below FLM, allow below-FLM remainder.
            rounded = np.where((positive_prob >= threshold) & (dc > 0) & (dc < flm), dc, rounded)
            rounded = np.where(rounded <= 0, np.nan, rounded)
            pred[idx] = rounded

    # Review optimizer: ranked allocation until each approximated DC pool is empty.
    ridx = np.where(review_mask)[0]
    if len(ridx):
        rwork = work.iloc[ridx].copy()
        need = row_need_score(rwork, conf[ridx], raw_packs_all[ridx])
        rwork["__idx"] = ridx
        rwork["__need"] = need
        # Approximate pool key using ONLY allowed columns.
        rwork["__pool"] = (
            rwork["Class Name"].astype(str).str.upper().str.strip() + "|" +
            rwork["Line Name"].astype(str).str.upper().str.strip() + "|" +
            rwork["Cost"].astype(str).str.upper().str.strip() + "|" +
            rwork["Dc Avail"].astype(str).str.upper().str.strip()
        )
        for _, sub in rwork.sort_values("__need", ascending=False).groupby("__pool", sort=False):
            # Initial pool from max Dc Avail in the group.
            remaining = float(np.nanmax(to_float_array(sub["Dc Avail"], 0.0)))
            sub_sorted = sub.sort_values("__need", ascending=False)
            for _, row in sub_sorted.iterrows():
                i = int(row["__idx"])
                if remaining <= 0: break
                if conf[i] < artifact_bundle["meta"].get("train_config", {}).get("review_threshold", 0.58):
                    continue
                flm = max(float(row["FLM"] if not pd.isna(row["FLM"]) else 1.0), 1.0)
                want = max(raw_packs_all[i] * flm, 0.0)
                if want <= 0: continue
                rounded = round(want / flm) * flm
                if rounded <= 0 and want > 0: rounded = flm
                if remaining < flm:
                    alloc = remaining  # below-FLM remainder allowed only at end.
                else:
                    alloc = min(rounded, remaining)
                    alloc = math.floor(alloc / flm) * flm
                    if alloc <= 0 and remaining > 0: alloc = min(flm, remaining)
                if alloc > 0:
                    pred[i] = alloc
                    remaining -= alloc

    out = work.copy()
    out["Predicted Final Alloc"] = pred
    out["Predicted Group"] = group_pred
    out["Allocation Confidence"] = conf
    out["Raw Predicted FLMs"] = raw_packs_all
    out["Predicted Final Alloc"] = out["Predicted Final Alloc"].where(~np.isnan(out["Predicted Final Alloc"]), "")
    return out
