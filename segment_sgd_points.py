from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import hstack, load_npz
from sklearn.metrics import mean_absolute_error


# -----------------------------
# Utils: file pairing (same as training)
# -----------------------------
_PART_RE = re.compile(r"part-(\d+)\.(parquet|npz)$")


def _part_num(p: Path) -> int:
    m = _PART_RE.search(p.name)
    if not m:
        raise ValueError(f"Cannot parse part number from: {p.name}")
    return int(m.group(1))


def pair_parts(base_dir: Path, items_dir: Optional[Path]) -> List[Tuple[Path, Optional[Path]]]:
    base_files = sorted(base_dir.glob("part-*.parquet"), key=_part_num)
    if not base_files:
        raise FileNotFoundError(f"No parquet parts found in {base_dir}")

    if items_dir is None:
        return [(bf, None) for bf in base_files]

    item_files = sorted(items_dir.glob("items-part-*.npz"), key=_part_num)
    if len(item_files) != len(base_files):
        raise ValueError(f"parts mismatch: parquet={len(base_files)} vs npz={len(item_files)}")

    pairs = []
    for bf, itf in zip(base_files, item_files):
        if _part_num(bf) != _part_num(itf):
            raise ValueError(f"part id mismatch: {bf.name} vs {itf.name}")
        pairs.append((bf, itf))
    return pairs


def load_all_parts(
    pairs: List[Tuple[Path, Optional[Path]]],
    max_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, Optional[sparse.csr_matrix]]:
    dfs: List[pd.DataFrame] = []
    mats: List[sparse.csr_matrix] = []

    total = 0
    for base_path, items_path in pairs:
        df = pd.read_parquet(base_path)

        if max_rows is not None and total + len(df) > max_rows:
            need = max_rows - total
            if need <= 0:
                break
            df = df.iloc[:need].copy()

        dfs.append(df)
        total += len(df)

        if items_path is not None:
            X = load_npz(items_path).tocsr()
            if X.shape[0] != len(df):
                X = X[: len(df)]
            mats.append(X)

        if max_rows is not None and total >= max_rows:
            break

    df_all = pd.concat(dfs, axis=0, ignore_index=True)

    X_items = None
    if mats:
        X_items = sparse.vstack(mats).tocsr()
        if X_items.shape[0] != len(df_all):
            raise ValueError(f"Row mismatch: df={len(df_all)} vs X_items={X_items.shape[0]}")
    return df_all, X_items


def _filter_df_and_items(
    df: pd.DataFrame,
    X_items: Optional[sparse.csr_matrix],
    mask: pd.Series,
) -> Tuple[pd.DataFrame, Optional[sparse.csr_matrix]]:
    idx = np.flatnonzero(mask.to_numpy())
    df2 = df.iloc[idx].reset_index(drop=True)
    if X_items is None:
        return df2, None
    return df2, X_items[idx]


# -----------------------------
# Derived features (MUST match training)
# -----------------------------

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    if "point_id" in df.columns and "hour" in df.columns:
        pid = pd.to_numeric(df["point_id"], errors="coerce").fillna(-1).astype("int32")
        hr = pd.to_numeric(df["hour"], errors="coerce").fillna(-1).astype("int16")
        df["point_hour"] = pid.astype(str) + "__" + hr.astype(str)

    if "items_qty_total_all" in df.columns:
        q = pd.to_numeric(df["items_qty_total_all"], errors="coerce").fillna(0.0).astype("float32")
        df["items_qty_total_all_log1p"] = np.log1p(q).astype("float32")
        df["items_qty_total_all_sq"] = (q * q).astype("float32")

    if "items_qty_total_all" in df.columns and "items_unique_all" in df.columns:
        q = pd.to_numeric(df["items_qty_total_all"], errors="coerce").fillna(0.0).astype("float32")
        u = pd.to_numeric(df["items_unique_all"], errors="coerce").fillna(0.0).astype("float32")
        df["items_qty_x_unique"] = (q * u).astype("float32")

    return df


# -----------------------------
# Split (MUST match training)
# -----------------------------

def split_by_point_order_id(
    df: pd.DataFrame,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    min_per_point: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "point_id" not in df.columns:
        raise ValueError("point_id is required for per-point split")

    train_idx: List[np.ndarray] = []
    val_idx: List[np.ndarray] = []
    test_idx: List[np.ndarray] = []

    for _, g in df.groupby("point_id", sort=False):
        g_sorted = g.sort_values("order_id")
        idx = g_sorted.index.to_numpy()
        n = len(idx)

        if n < min_per_point:
            train_idx.append(idx)
            continue

        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        n_train = max(n_train, 1)
        n_val = max(n_val, 1)

        n_test = n - n_train - n_val
        if n_test < 1:
            n_val = max(0, n - n_train - 1)
            n_test = n - n_train - n_val

        train_idx.append(idx[:n_train])
        if n_val > 0:
            val_idx.append(idx[n_train:n_train + n_val])
        test_idx.append(idx[n_train + n_val:])

    train = np.concatenate(train_idx) if train_idx else np.array([], dtype=int)
    val = np.concatenate(val_idx) if val_idx else np.array([], dtype=int)
    test = np.concatenate(test_idx) if test_idx else np.array([], dtype=int)
    return train, val, test


# -----------------------------
# Baseline (point-hour mean) for comparison / blending
# -----------------------------

class PointHourMeanBaseline:
    def __init__(self):
        self.global_mean_: float = 0.0
        self.point_mean_: Dict[int, float] = {}
        self.ph_mean_: Dict[Tuple[int, int, int], float] = {}

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "PointHourMeanBaseline":
        self.global_mean_ = float(np.mean(y))

        if "point_id" not in df.columns:
            return self

        tmp = df[["point_id", "weekday", "hour"]].copy()
        tmp["y"] = y

        pm = tmp.groupby("point_id")["y"].mean()
        self.point_mean_ = {int(k): float(v) for k, v in pm.items()}

        ph = tmp.groupby(["point_id", "weekday", "hour"])["y"].mean()
        self.ph_mean_ = {(int(a), int(b), int(c)): float(v) for (a, b, c), v in ph.items()}
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(df), dtype=float)
        for i, row in enumerate(df.itertuples(index=False)):
            pid = getattr(row, "point_id", None)
            wd = getattr(row, "weekday", None)
            hr = getattr(row, "hour", None)

            if pid is None or wd is None or hr is None:
                out[i] = self.global_mean_
                continue

            key = (int(pid), int(wd), int(hr))
            if key in self.ph_mean_:
                out[i] = self.ph_mean_[key]
            else:
                out[i] = self.point_mean_.get(int(pid), self.global_mean_)
        return out


# -----------------------------
# Segment metrics
# -----------------------------
@dataclass
class SegMetrics:
    count: int
    mae: float
    rmse: float
    bias: float
    p50_abs_err: float
    p90_abs_err: float
    tail_30m: float
    tail_45m: float


def segment_metrics(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_cols: List[str],
    min_count: int = 200,
) -> pd.DataFrame:
    work = df[group_cols].copy()
    work["_y"] = y_true
    work["_p"] = y_pred
    work["_err"] = work["_y"] - work["_p"]
    work["_abs"] = work["_err"].abs()
    work["_sq"] = (work["_err"] ** 2)

    g = work.groupby(group_cols, dropna=False)

    agg = g.agg(
        count=("_y", "size"),
        mae=("_abs", "mean"),
        bias=("_err", "mean"),
        mse=("_sq", "mean"),
    ).reset_index()

    agg["rmse"] = np.sqrt(agg["mse"])
    agg = agg.drop(columns=["mse"])

    q = g["_abs"].quantile([0.5, 0.9]).unstack(level=-1).reset_index()
    q = q.rename(columns={0.5: "p50_abs_err", 0.9: "p90_abs_err"})
    agg = agg.merge(q, on=group_cols, how="left")

    tail = g["_abs"].apply(lambda s: pd.Series({
        "tail_30m": float((s > 30).mean()),
        "tail_45m": float((s > 45).mean()),
    })).reset_index()
    agg = agg.merge(tail, on=group_cols, how="left")

    agg = agg[agg["count"] >= min_count].copy()
    agg = agg.sort_values(["p90_abs_err", "mae"], ascending=False)
    return agg


# -----------------------------
# Plotting (one figure per plot)
# -----------------------------
def plot_top_bar(df_seg: pd.DataFrame, x_col: str, y_col: str, title: str, out_path: Path, top_n: int = 20):
    dfp = df_seg.head(top_n).copy()
    plt.figure()
    plt.bar(dfp[x_col].astype(str), dfp[y_col].to_numpy())
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(y_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_line_by_bucket(df_seg: pd.DataFrame, x_col: str, y_col: str, title: str, out_path: Path):
    dfp = df_seg.sort_values(x_col).copy()
    plt.figure()
    plt.plot(dfp[x_col].to_numpy(), dfp[y_col].to_numpy(), marker="o")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_scatter(x: np.ndarray, y: np.ndarray, title: str, out_path: Path, xlab: str, ylab: str, sample_n: int = 40000):
    n = len(x)
    if n > sample_n:
        idx = np.random.RandomState(42).choice(n, size=sample_n, replace=False)
        x = x[idx]
        y = y[idx]
    plt.figure()
    plt.scatter(x, y, s=4, alpha=0.3)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--items-dir", default=None)
    ap.add_argument("--model-dir", required=True, help="directory with preprocessor.joblib + ridge.joblib + sgd.joblib (+ run_info.json)")
    ap.add_argument("--out-dir", default="diag_out")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--min-count", type=int, default=200)

    ap.add_argument("--use-log-target", action="store_true", help="force log1p inversion (if run_info not found)")

    ap.add_argument("--min-per-point", type=int, default=50)
    ap.add_argument("--train-frac", type=float, default=0.80)
    ap.add_argument("--val-frac", type=float, default=0.10)

    ap.add_argument("--drop-preorders", action="store_true")
    ap.add_argument("--exclude-points", type=str, default=None)

    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    items_dir = Path(args.items_dir) if args.items_dir else None
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_info = {}
    run_info_path = model_dir / "run_info.json"
    if run_info_path.exists():
        run_info = json.loads(run_info_path.read_text(encoding="utf-8"))

    use_log_target = bool(args.use_log_target) or bool(run_info.get("use_log_target", False))

    cap_minutes = None
    pred_cap_info = run_info.get("pred_cap") or {}
    if pred_cap_info.get("enabled"):
        cap_minutes = pred_cap_info.get("minutes")

    blend_info = run_info.get("blend") or {}
    w_ridge = None
    w_sgd = None
    if blend_info.get("enabled"):
        weights = blend_info.get("weights") or {}
        w_ridge = (weights.get("ridge") or {}).get("w_model")
        w_sgd = (weights.get("sgd") or {}).get("w_model")

    pairs = pair_parts(base_dir, items_dir)
    df, X_items = load_all_parts(pairs, max_rows=args.max_rows)

    if "delivery_at_client" in df.columns:
        df["delivery_at_client"] = pd.to_numeric(df["delivery_at_client"], errors="coerce").fillna(0).astype("int8")
        if args.drop_preorders:
            mask = df["delivery_at_client"] == 0
            df, X_items = _filter_df_and_items(df, X_items, mask)

    if args.exclude_points and "point_id" in df.columns:
        excluded_points = [int(x) for x in args.exclude_points.split(",") if x.strip()]
        if excluded_points:
            mask = ~df["point_id"].isin(excluded_points)
            df, X_items = _filter_df_and_items(df, X_items, mask)

    df = add_derived_features(df)

    train_idx, _, test_idx = split_by_point_order_id(
        df,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        min_per_point=args.min_per_point,
    )
    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()

    y_train = pd.to_numeric(df_train["cooking_minutes"], errors="coerce").to_numpy().astype(float)
    y_test = pd.to_numeric(df_test["cooking_minutes"], errors="coerce").to_numpy().astype(float)

    X_items_train = X_items[train_idx] if X_items is not None else None
    X_items_test = X_items[test_idx] if X_items is not None else None

    pre = joblib.load(model_dir / "preprocessor.joblib")
    ridge = joblib.load(model_dir / "ridge.joblib")
    sgd = joblib.load(model_dir / "sgd.joblib")

    baseline_path = model_dir / "baseline_point_hour_mean.joblib"
    if baseline_path.exists():
        baseline = joblib.load(baseline_path)
    else:
        baseline = PointHourMeanBaseline().fit(df_train, y_train)

    X_train_tab = pre.transform(df_train)
    X_test_tab = pre.transform(df_test)

    X_train = hstack([X_train_tab, X_items_train], format="csr") if X_items_train is not None else X_train_tab.tocsr()
    X_test = hstack([X_test_tab, X_items_test], format="csr") if X_items_test is not None else X_test_tab.tocsr()

    def pred_to_minutes(p: np.ndarray) -> np.ndarray:
        if use_log_target:
            p = np.expm1(p)
        p = np.clip(p, 0.0, None)
        if cap_minutes is not None:
            p = np.minimum(p, float(cap_minutes))
        return p

    pred_ridge = pred_to_minutes(ridge.predict(X_test))
    pred_sgd = pred_to_minutes(sgd.predict(X_test))
    pred_ph = baseline.predict(df_test)

    models: Dict[str, np.ndarray] = {
        "sgd": pred_sgd,
        "ridge": pred_ridge,
        "baseline_point_hour_mean": pred_ph,
    }

    if w_ridge is not None:
        models["ridge_blend"] = float(w_ridge) * pred_ridge + (1.0 - float(w_ridge)) * pred_ph
    if w_sgd is not None:
        models["sgd_blend"] = float(w_sgd) * pred_sgd + (1.0 - float(w_sgd)) * pred_ph

    overall_rows = []
    for name, yp in models.items():
        mae = float(mean_absolute_error(y_test, yp))
        p90 = float(np.quantile(np.abs(y_test - yp), 0.90))
        tail45 = float((np.abs(y_test - yp) > 45).mean())
        overall_rows.append({"model": name, "MAE": mae, "P90_abs_err": p90, "tail_45m": tail45})
    overall = pd.DataFrame(overall_rows).sort_values("MAE")
    overall.to_csv(out_dir / "overall_summary.csv", index=False)

    for name, yp in models.items():
        abs_err = np.abs(y_test - yp)
        top_idx = np.argsort(-abs_err)[:200]
        worst = df_test.iloc[top_idx].copy()
        worst["y_true"] = y_test[top_idx]
        worst["y_pred"] = yp[top_idx]
        worst["abs_err"] = abs_err[top_idx]
        worst["err"] = (worst["y_true"] - worst["y_pred"])
        keep = [c for c in worst.columns if c in [
            "order_id", "point_id", "city_id", "delivery_method", "delivery_at_client",
            "persons", "weekday", "hour", "is_weekend",
            "items_qty_total_all", "items_lines_all", "items_unique_all",
            "items_qty_total_all_log1p", "items_qty_total_all_sq", "items_qty_x_unique",
            "point_hour",
            "y_true", "y_pred", "abs_err", "err",
        ]]
        worst[keep].to_csv(out_dir / f"worst_orders__{name}.csv", index=False)

    segment_defs: List[Tuple[List[str], str]] = [
        (["point_id"], "point"),
        (["hour"], "hour"),
        (["weekday"], "weekday"),
        (["delivery_method"], "delivery_method"),
        (["point_hour"], "point_hour_token"),
        (["point_id", "hour"], "point_hour_pair"),
    ]

    for col in ["items_qty_total_all", "items_unique_all"]:
        if col in df_test.columns:
            vals = pd.to_numeric(df_test[col], errors="coerce").fillna(0.0).to_numpy()
            qs = np.unique(np.quantile(vals, np.linspace(0, 1, 11)))
            if len(qs) >= 3:
                df_test[f"{col}_decile"] = np.digitize(vals, qs[1:-1], right=True)
                segment_defs.append(([f"{col}_decile"], f"{col}_decile"))

    for model_name, yp in models.items():
        for cols, tag in segment_defs:
            if any(c not in df_test.columns for c in cols):
                continue

            seg = segment_metrics(df_test, y_test, yp, cols, min_count=args.min_count)
            seg_path = out_dir / f"segments__{model_name}__{tag}.csv"
            seg.to_csv(seg_path, index=False)

            if len(cols) == 1:
                x_col = cols[0]
                if x_col in {"point_id", "city_id"}:
                    plot_top_bar(seg, x_col, "p90_abs_err",
                                 f"P90 abs error by {x_col} ({model_name})",
                                 out_dir / f"p90_top__{model_name}__{x_col}.png", top_n=20)
                    plot_top_bar(seg, x_col, "mae",
                                 f"MAE by {x_col} ({model_name})",
                                 out_dir / f"mae_top__{model_name}__{x_col}.png", top_n=20)
                elif x_col in {"hour", "weekday"} or x_col.endswith("_decile"):
                    plot_line_by_bucket(seg, x_col, "mae",
                                        f"MAE by {x_col} ({model_name})",
                                        out_dir / f"mae_line__{model_name}__{x_col}.png")
                    plot_line_by_bucket(seg, x_col, "p90_abs_err",
                                        f"P90 abs err by {x_col} ({model_name})",
                                        out_dir / f"p90_line__{model_name}__{x_col}.png")
                else:
                    plot_top_bar(seg, x_col, "p90_abs_err",
                                 f"P90 abs error by {x_col} ({model_name})",
                                 out_dir / f"p90_top__{model_name}__{x_col}.png", top_n=20)

        if "items_qty_total_all" in df_test.columns:
            plot_scatter(
                x=pd.to_numeric(df_test["items_qty_total_all"], errors="coerce").fillna(0.0).to_numpy(),
                y=np.abs(y_test - yp),
                title=f"Abs error vs items_qty_total_all ({model_name})",
                out_path=out_dir / f"scatter_abs_err__{model_name}__items_qty_total_all.png",
                xlab="items_qty_total_all",
                ylab="abs error (minutes)",
            )
        if "items_unique_all" in df_test.columns:
            plot_scatter(
                x=pd.to_numeric(df_test["items_unique_all"], errors="coerce").fillna(0.0).to_numpy(),
                y=np.abs(y_test - yp),
                title=f"Abs error vs items_unique_all ({model_name})",
                out_path=out_dir / f"scatter_abs_err__{model_name}__items_unique_all.png",
                xlab="items_unique_all",
                ylab="abs error (minutes)",
            )

    (out_dir / "diag_run_info.json").write_text(json.dumps({
        "use_log_target": use_log_target,
        "cap_minutes": cap_minutes,
        "blend_weights": {"ridge": w_ridge, "sgd": w_sgd},
        "split": {"train_frac": args.train_frac, "val_frac": args.val_frac, "min_per_point": args.min_per_point},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Outputs in: {out_dir}")
    print("Start with: overall_summary.csv and segments__sgd__point.csv (sorted by worst P90).")


if __name__ == "__main__":
    main()
