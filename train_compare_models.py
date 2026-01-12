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

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# Utils: file pairing
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

    pairs: List[Tuple[Path, Optional[Path]]] = []
    for bf, itf in zip(base_files, item_files):
        if _part_num(bf) != _part_num(itf):
            raise ValueError(f"part id mismatch: {bf.name} vs {itf.name}")
        pairs.append((bf, itf))
    return pairs


# -----------------------------
# Split: per-point time-aware (order_id as proxy time)
# -----------------------------

def split_by_point_order_id(
    df: pd.DataFrame,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    min_per_point: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Гарантирует, что каждая точка (point_id) отдаёт ~train_frac заказов в train (по времени),
    сортируя внутри точки по order_id (proxy времени).

    Если у точки мало заказов (< min_per_point), то все заказы точки отправляются в train.
    """
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
# Metrics & plotting
# -----------------------------

@dataclass
class Metrics:
    mae: float
    rmse: float
    r2: float
    mape: float
    p50_abs_err: float
    p90_abs_err: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    denom = np.maximum(y_true, 1.0)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

    abs_err = np.abs(y_true - y_pred)
    p50 = float(np.quantile(abs_err, 0.50))
    p90 = float(np.quantile(abs_err, 0.90))

    return Metrics(mae=mae, rmse=rmse, r2=r2, mape=mape, p50_abs_err=p50, p90_abs_err=p90)


def plot_model_bars(metrics_df: pd.DataFrame, out_path: Path, metric_col: str = "MAE") -> None:
    plt.figure()
    x = np.arange(len(metrics_df))
    plt.bar(x, metrics_df[metric_col].to_numpy())
    plt.xticks(x, metrics_df["model"].tolist(), rotation=25, ha="right")
    plt.ylabel(metric_col)
    plt.title(f"{metric_col} by model")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_pred_vs_true(y_true: np.ndarray, preds: Dict[str, np.ndarray], out_path: Path, sample_n: int = 20000) -> None:
    n = len(y_true)
    if n > sample_n:
        idx = np.random.RandomState(42).choice(n, size=sample_n, replace=False)
        yt = y_true[idx]
        preds_s = {k: v[idx] for k, v in preds.items()}
    else:
        yt = y_true
        preds_s = preds

    for name, yp in preds_s.items():
        p = out_path.parent / f"{out_path.stem}__{name}.png"
        plt.figure()
        plt.scatter(yt, yp, s=4, alpha=0.3)
        plt.xlabel("True cooking_minutes")
        plt.ylabel("Predicted cooking_minutes")
        plt.title(f"Pred vs True: {name}")
        plt.tight_layout()
        plt.savefig(p, dpi=160)
        plt.close()


def plot_residual_hist(y_true: np.ndarray, preds: Dict[str, np.ndarray], out_dir: Path) -> None:
    for name, yp in preds.items():
        res = (y_true - yp).astype(float)
        p = out_dir / f"residual_hist__{name}.png"
        plt.figure()
        plt.hist(res, bins=80)
        plt.xlabel("Residual (true - pred), minutes")
        plt.ylabel("Count")
        plt.title(f"Residual histogram: {name}")
        plt.tight_layout()
        plt.savefig(p, dpi=160)
        plt.close()


def plot_calibration_deciles(y_true: np.ndarray, preds: Dict[str, np.ndarray], out_dir: Path) -> None:
    for name, yp in preds.items():
        yp = yp.astype(float)
        y_true_f = y_true.astype(float)

        qs = np.quantile(yp, np.linspace(0, 1, 11))
        qs = np.unique(qs)
        if len(qs) < 3:
            continue

        bins = np.digitize(yp, qs[1:-1], right=True)
        mean_true, mean_pred = [], []
        for b in range(bins.min(), bins.max() + 1):
            mask = bins == b
            if mask.sum() == 0:
                continue
            mean_true.append(y_true_f[mask].mean())
            mean_pred.append(yp[mask].mean())

        p = out_dir / f"calibration_deciles__{name}.png"
        plt.figure()
        plt.plot(mean_pred, mean_true, marker="o")
        plt.xlabel("Mean predicted (bin)")
        plt.ylabel("Mean true (bin)")
        plt.title(f"Calibration (deciles): {name}")
        plt.tight_layout()
        plt.savefig(p, dpi=160)
        plt.close()


# -----------------------------
# Data loading
# -----------------------------

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
# Feature preprocessing
# -----------------------------

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляем:
      - point_hour: категориальная фича = "{point_id}__{hour}"
      - qty_log1p = log1p(items_qty_total_all)
      - qty_sq = items_qty_total_all^2
      - qty_x_unique = items_qty_total_all * items_unique_all
    """
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


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in ["point_id", "city_id", "delivery_method", "point_hour"] if c in df.columns]
    num_cols = [c for c in [
        "weekday", "hour", "is_weekend",
        "items_qty_total_all", "items_lines_all", "items_unique_all",
        "items_qty_total_all_log1p", "items_qty_total_all_sq", "items_qty_x_unique",
    ] if c in df.columns]

    try:
        cat_tf = OneHotEncoder(handle_unknown="ignore", min_frequency=50)
    except TypeError:
        cat_tf = OneHotEncoder(handle_unknown="ignore")

    num_tf = StandardScaler(with_mean=False)

    return ColumnTransformer(
        transformers=[
            ("cat", cat_tf, cat_cols),
            ("num", num_tf, num_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )


def make_X(
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
    X_items: Optional[sparse.csr_matrix],
    fit: bool,
) -> sparse.csr_matrix:
    X_tab = preprocessor.fit_transform(df) if fit else preprocessor.transform(df)
    if X_items is None:
        return X_tab.tocsr()
    return hstack([X_tab, X_items], format="csr")


# -----------------------------
# Baselines
# -----------------------------

class GlobalMedianBaseline:
    def __init__(self):
        self.median_: float = 0.0

    def fit(self, y: np.ndarray) -> "GlobalMedianBaseline":
        self.median_ = float(np.median(y))
        return self

    def predict(self, n: int) -> np.ndarray:
        return np.full(shape=(n,), fill_value=self.median_, dtype=float)


class PointHourMeanBaseline:
    """
    mean y per (point_id, weekday, hour). fallback -> point mean -> global mean
    """
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
# Blend helper
# -----------------------------

def _metric_value(m: Metrics, metric: str) -> float:
    if metric == "p90":
        return m.p90_abs_err
    if metric == "mae":
        return m.mae
    if metric == "rmse":
        return m.rmse
    raise ValueError(f"Unknown blend metric: {metric}")


def best_blend_weight(
    y_val: np.ndarray,
    pred_model: np.ndarray,
    pred_base: np.ndarray,
    metric: str = "p90",
    grid: Optional[np.ndarray] = None,
) -> Tuple[float, Metrics]:
    if grid is None:
        grid = np.linspace(0.0, 1.0, 21)

    best_w = 1.0
    best_m = compute_metrics(y_val, pred_model)
    best_score = _metric_value(best_m, metric)

    for w in grid:
        p = w * pred_model + (1.0 - w) * pred_base
        m = compute_metrics(y_val, p)
        score = _metric_value(m, metric)

        if (score < best_score) or (score == best_score and m.mae < best_m.mae):
            best_w = float(w)
            best_m = m
            best_score = score

    return best_w, best_m


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True, help="prepared_base directory with part-*.parquet")
    ap.add_argument("--items-dir", default=None, help="prepared_items_hash directory with items-part-*.npz")
    ap.add_argument("--out-dir", default="model_out", help="output directory for metrics/plots/models")
    ap.add_argument("--max-rows", type=int, default=None, help="optional cap for quick experiments")

    ap.add_argument("--use-log-target", action="store_true", help="train on log1p(y) and invert for metrics")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--min-per-point", type=int, default=50)
    ap.add_argument("--train-frac", type=float, default=0.80)
    ap.add_argument("--val-frac", type=float, default=0.10)

    ap.add_argument("--drop-preorders", action="store_true", help="Drop delivery_at_client=1 if column exists")
    ap.add_argument("--exclude-points", type=str, default=None, help="Comma-separated point_ids to drop, e.g. 624,777")

    # cap predictions
    ap.add_argument("--no-cap", action="store_true", help="Disable prediction capping")
    ap.add_argument("--pred-cap", type=float, default=None, help="Hard cap on predictions in minutes (e.g. 240)")
    ap.add_argument("--pred-cap-quantile", type=float, default=0.999,
                    help="If pred-cap not set: cap = quantile(y_train, q). Set --no-cap to disable.")

    # models
    ap.add_argument("--ridge-alpha", type=float, default=10.0)

    ap.add_argument("--sgd-alpha", type=float, default=1e-5)
    ap.add_argument("--sgd-loss", type=str, default="huber", choices=["huber", "squared_error"])
    ap.add_argument("--sgd-epsilon", type=float, default=1.35)
    ap.add_argument("--sgd-max-iter", type=int, default=50)

    # blend
    ap.add_argument("--no-blend", action="store_true", help="Disable blending with baseline_point_hour_mean")
    ap.add_argument("--blend-metric", type=str, default="p90", choices=["p90", "mae", "rmse"])

    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    items_dir = Path(args.items_dir) if args.items_dir else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = pair_parts(base_dir, items_dir)
    df, X_items = load_all_parts(pairs, max_rows=args.max_rows)

    preorder_share = None
    if "delivery_at_client" in df.columns:
        df["delivery_at_client"] = pd.to_numeric(df["delivery_at_client"], errors="coerce").fillna(0).astype("int8")
        preorder_share = float((df["delivery_at_client"] == 1).mean())
        if args.drop_preorders:
            mask = df["delivery_at_client"] == 0
            df, X_items = _filter_df_and_items(df, X_items, mask)

    excluded_points: List[int] = []
    if args.exclude_points and "point_id" in df.columns:
        excluded_points = [int(x) for x in args.exclude_points.split(",") if x.strip()]
        if excluded_points:
            mask = ~df["point_id"].isin(excluded_points)
            df, X_items = _filter_df_and_items(df, X_items, mask)

    df = add_derived_features(df)

    y = pd.to_numeric(df["cooking_minutes"], errors="coerce").to_numpy().astype(float)

    train_idx, val_idx, test_idx = split_by_point_order_id(
        df,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        min_per_point=args.min_per_point,
    )

    df_train, df_val, df_test = df.iloc[train_idx].copy(), df.iloc[val_idx].copy(), df.iloc[test_idx].copy()
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    X_items_train = X_items[train_idx] if X_items is not None else None
    X_items_val = X_items[val_idx] if X_items is not None else None
    X_items_test = X_items[test_idx] if X_items is not None else None

    pre = build_preprocessor(df_train)
    X_train = make_X(pre, df_train, X_items_train, fit=True)
    X_val = make_X(pre, df_val, X_items_val, fit=False)
    X_test = make_X(pre, df_test, X_items_test, fit=False)

    def y_to_train_space(y_arr: np.ndarray) -> np.ndarray:
        return np.log1p(y_arr) if args.use_log_target else y_arr

    y_train_t = y_to_train_space(y_train)

    # cap by train quantile / hard
    cap_mode = "disabled"
    cap_minutes: Optional[float] = None
    if not args.no_cap:
        if args.pred_cap is not None:
            cap_mode = "hard"
            cap_minutes = float(args.pred_cap)
        else:
            q = float(args.pred_cap_quantile)
            if 0.0 < q < 1.0:
                cap_mode = "quantile"
                cap_minutes = float(np.quantile(y_train, q))

    def pred_to_minutes(p: np.ndarray) -> np.ndarray:
        if args.use_log_target:
            p = np.expm1(p)
        p = np.clip(p, 0.0, None)
        if cap_minutes is not None:
            p = np.minimum(p, cap_minutes)
        return p

    # ---------------- Baselines ----------------
    results = []
    preds_test: Dict[str, np.ndarray] = {}

    m_median = GlobalMedianBaseline().fit(y_train)
    pred = m_median.predict(len(y_test))
    preds_test["baseline_median"] = pred
    results.append(("baseline_median", compute_metrics(y_test, pred)))

    baseline = PointHourMeanBaseline().fit(df_train, y_train)
    pred_base_val = baseline.predict(df_val)
    pred_base_test = baseline.predict(df_test)
    preds_test["baseline_point_hour_mean"] = pred_base_test
    results.append(("baseline_point_hour_mean", compute_metrics(y_test, pred_base_test)))

    # ---------------- Models ----------------
    ridge = Ridge(alpha=args.ridge_alpha, solver="sag", random_state=args.seed)
    ridge.fit(X_train, y_train_t)
    pred_ridge_val = pred_to_minutes(ridge.predict(X_val))
    pred_ridge_test = pred_to_minutes(ridge.predict(X_test))
    preds_test["ridge"] = pred_ridge_test
    results.append(("ridge", compute_metrics(y_test, pred_ridge_test)))

    sgd = SGDRegressor(
        loss=args.sgd_loss,
        alpha=args.sgd_alpha,
        epsilon=args.sgd_epsilon if args.sgd_loss == "huber" else 0.1,
        penalty="l2",
        learning_rate="invscaling",
        eta0=0.01,
        power_t=0.25,
        max_iter=args.sgd_max_iter,
        tol=1e-4,
        random_state=args.seed,
    )
    sgd.fit(X_train, y_train_t)
    pred_sgd_val = pred_to_minutes(sgd.predict(X_val))
    pred_sgd_test = pred_to_minutes(sgd.predict(X_test))
    preds_test["sgd"] = pred_sgd_test
    results.append(("sgd", compute_metrics(y_test, pred_sgd_test)))

    # ---------------- Blend with baseline ----------------
    blend_info = {"enabled": False}
    if not args.no_blend and len(val_idx) > 0:
        blend_info["enabled"] = True
        blend_info["metric"] = args.blend_metric
        blend_info["grid"] = {"n": 21, "from": 0.0, "to": 1.0}

        w_ridge, m_ridge_val = best_blend_weight(y_val, pred_ridge_val, pred_base_val, metric=args.blend_metric)
        pred_ridge_blend = w_ridge * pred_ridge_test + (1.0 - w_ridge) * pred_base_test
        preds_test["ridge_blend"] = pred_ridge_blend
        results.append(("ridge_blend", compute_metrics(y_test, pred_ridge_blend)))

        w_sgd, m_sgd_val = best_blend_weight(y_val, pred_sgd_val, pred_base_val, metric=args.blend_metric)
        pred_sgd_blend = w_sgd * pred_sgd_test + (1.0 - w_sgd) * pred_base_test
        preds_test["sgd_blend"] = pred_sgd_blend
        results.append(("sgd_blend", compute_metrics(y_test, pred_sgd_blend)))

        blend_info["weights"] = {
            "ridge": {"w_model": w_ridge, "val_metrics": m_ridge_val.__dict__},
            "sgd": {"w_model": w_sgd, "val_metrics": m_sgd_val.__dict__},
        }

    # ---------------- Save metrics ----------------
    rows = []
    for name, met in results:
        rows.append({
            "model": name,
            "MAE": met.mae,
            "RMSE": met.rmse,
            "R2": met.r2,
            "MAPE_%": met.mape,
            "P50_abs_err": met.p50_abs_err,
            "P90_abs_err": met.p90_abs_err,
        })
    metrics_df = pd.DataFrame(rows).sort_values("MAE", ascending=True)
    metrics_path = out_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # ---------------- Plots ----------------
    plot_model_bars(metrics_df, out_dir / "mae_by_model.png", metric_col="MAE")
    plot_model_bars(metrics_df, out_dir / "p90_by_model.png", metric_col="P90_abs_err")
    plot_pred_vs_true(y_test, preds_test, out_dir / "pred_vs_true.png", sample_n=20000)
    plot_residual_hist(y_test, preds_test, out_dir)
    plot_calibration_deciles(y_test, preds_test, out_dir)

    # ---------------- Save models ----------------
    joblib.dump(pre, out_dir / "preprocessor.joblib")
    joblib.dump(ridge, out_dir / "ridge.joblib")
    joblib.dump(sgd, out_dir / "sgd.joblib")
    joblib.dump(baseline, out_dir / "baseline_point_hour_mean.joblib")

    points_total = int(df["point_id"].nunique()) if "point_id" in df.columns else None
    points_in_test = int(df_test["point_id"].nunique()) if "point_id" in df_test.columns else None

    info = {
        "seed": int(args.seed),
        "use_log_target": bool(args.use_log_target),
        "preorders": {
            "drop_preorders_flag": bool(args.drop_preorders),
            "preorder_share_before_drop": preorder_share,
        },
        "excluded_points": excluded_points,
        "split": {
            "type": "per_point_order_id",
            "train_frac": args.train_frac,
            "val_frac": args.val_frac,
            "min_per_point": args.min_per_point,
            "rows": {"train": int(len(df_train)), "val": int(len(df_val)), "test": int(len(df_test))},
            "points_total": points_total,
            "points_in_test": points_in_test,
        },
        "pred_cap": {
            "enabled": cap_minutes is not None,
            "mode": cap_mode,
            "minutes": cap_minutes,
            "quantile": None if args.no_cap or args.pred_cap is not None else float(args.pred_cap_quantile),
        },
        "ridge": {"alpha": args.ridge_alpha},
        "sgd": {
            "alpha": args.sgd_alpha,
            "loss": args.sgd_loss,
            "epsilon": args.sgd_epsilon,
            "max_iter": args.sgd_max_iter,
        },
        "blend": blend_info,
        "files": {
            "metrics": str(metrics_path),
            "plots_dir": str(out_dir),
            "models": {
                "preprocessor": "preprocessor.joblib",
                "ridge": "ridge.joblib",
                "sgd": "sgd.joblib",
                "baseline_point_hour_mean": "baseline_point_hour_mean.joblib",
            },
        },
        "notes": "X = [OHE(cats)+Scaled(nums)] + hashed_items_sparse (if provided). Split is per-point by order_id.",
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(metrics_df.to_string(index=False))
    print("\nSplit info:")
    print(json.dumps(info["split"], ensure_ascii=False, indent=2))
    if preorder_share is not None:
        print(f"\nPreorders share in loaded data (before optional drop): {preorder_share:.4f}")
    if cap_minutes is not None:
        print(f"\nPrediction cap: mode={cap_mode}, minutes={cap_minutes:.2f}")
    if blend_info.get("enabled"):
        w_r = blend_info["weights"]["ridge"]["w_model"]
        w_s = blend_info["weights"]["sgd"]["w_model"]
        print(f"\nBlend enabled (metric={args.blend_metric}): ridge w={w_r:.2f}, sgd w={w_s:.2f}")
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
