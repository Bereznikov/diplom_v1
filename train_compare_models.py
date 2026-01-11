from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import timezone, datetime
from pathlib import Path
from random import randint
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import hstack
from scipy.sparse import load_npz
import matplotlib.pyplot as plt

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

    pairs = []
    for bf, itf in zip(base_files, item_files):
        if _part_num(bf) != _part_num(itf):
            raise ValueError(f"part id mismatch: {bf.name} vs {itf.name}")
        pairs.append((bf, itf))
    return pairs


# -----------------------------
# Split: order_id-based (proxy time)
# -----------------------------

def split_by_order_id(df: pd.DataFrame, train_q: float = 0.80, val_q: float = 0.90) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits indices by order_id quantiles. Assumption: order_id roughly increases over time.
    """
    oid = df["order_id"].to_numpy()
    q_train = np.quantile(oid, train_q)
    q_val = np.quantile(oid, val_q)

    train_idx = np.where(oid <= q_train)[0]
    val_idx = np.where((oid > q_train) & (oid <= q_val))[0]
    test_idx = np.where(oid > q_val)[0]
    return train_idx, val_idx, test_idx


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

    denom = np.maximum(y_true, 1.0)  # чтобы не взрывалось на малых значениях
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

    abs_err = np.abs(y_true - y_pred)
    p50 = float(np.quantile(abs_err, 0.50))
    p90 = float(np.quantile(abs_err, 0.90))

    return Metrics(mae=mae, rmse=rmse, r2=r2, mape=mape, p50_abs_err=p50, p90_abs_err=p90)

def plot_model_bars(metrics_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure()
    x = np.arange(len(metrics_df))
    plt.bar(x, metrics_df["MAE"].to_numpy())
    plt.xticks(x, metrics_df["model"].tolist(), rotation=25, ha="right")
    plt.ylabel("MAE (minutes)")
    plt.title("MAE by model")
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

    # отдельный график на модель (требование “один график — одна фигура”)
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
    """
    Разбиваем по децилям предсказания: mean(true) vs mean(pred) по корзинам.
    """
    for name, yp in preds.items():
        p = out_dir / f"calibration_deciles__{name}.png"
        yp = yp.astype(float)
        y_true_f = y_true.astype(float)

        # корзины по предикту
        qs = np.quantile(yp, np.linspace(0, 1, 11))
        # защита от одинаковых квантилей
        qs = np.unique(qs)
        if len(qs) < 3:
            continue

        bins = np.digitize(yp, qs[1:-1], right=True)
        mean_true = []
        mean_pred = []
        for b in range(bins.min(), bins.max() + 1):
            mask = bins == b
            if mask.sum() == 0:
                continue
            mean_true.append(y_true_f[mask].mean())
            mean_pred.append(yp[mask].mean())

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
            if max_rows is not None and X.shape[0] > len(df):
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


# -----------------------------
# Feature preprocessing
# -----------------------------

def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Табличные признаки:
      - категориальные: point_id, city_id, delivery_method
      - числовые: delivery_at_client, persons, weekday, hour, is_weekend, items_* агрегаты
    """
    cat_cols = []
    for c in ["point_id", "city_id", "delivery_method"]:
        if c in df.columns:
            cat_cols.append(c)

    num_cols = []
    for c in [
        "delivery_at_client", "persons", "weekday", "hour", "is_weekend",
        "items_qty_total_all", "items_lines_all", "items_unique_all"
    ]:
        if c in df.columns:
            num_cols.append(c)

    # OneHotEncoder -> sparse
    cat_tf = OneHotEncoder(handle_unknown="ignore")

    # scaler for numeric; with_mean=False to keep sparse compatibility
    num_tf = StandardScaler(with_mean=False)

    return ColumnTransformer(
        transformers=[
            ("cat", cat_tf, cat_cols),
            ("num", num_tf, num_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

def make_X(preprocessor: ColumnTransformer, df: pd.DataFrame, X_items: Optional[sparse.csr_matrix], fit: bool) -> sparse.csr_matrix:
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

        # point mean
        pm = tmp.groupby("point_id")["y"].mean()
        self.point_mean_ = {int(k): float(v) for k, v in pm.items()}

        # point-hour mean
        ph = tmp.groupby(["point_id", "weekday", "hour"])["y"].mean()
        self.ph_mean_ = {(int(a), int(b), int(c)): float(v) for (a, b, c), v in ph.items()}
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(df), dtype=float)
        for i, row in enumerate(df.itertuples(index=False)):
            # getattr for safety
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
# Training / evaluation
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True, help="prepared_base directory with part-*.parquet")
    ap.add_argument("--items-dir", default=None, help="prepared_items_hash directory with items-part-*.npz")
    ap.add_argument("--out-dir", default="model_out", help="output directory for metrics/plots/models")
    ap.add_argument("--max-rows", type=int, default=None, help="optional cap for quick experiments")
    ap.add_argument("--use-log-target", action="store_true", help="train on log1p(y) and invert for metrics")
    ap.add_argument("--seed", type=int, default=randint(20, 100000))
    ap.add_argument("--ridge-alpha", type=float, default=10.0)
    ap.add_argument("--sgd-alpha", type=float, default=1e-5)
    ap.add_argument("--sgd-loss", type=str, default="huber", choices=["huber", "squared_error"])
    ap.add_argument("--sgd-epsilon", type=float, default=1.35)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)

    base_dir = Path(args.base_dir)
    items_dir = Path(args.items_dir) if args.items_dir else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = pair_parts(base_dir, items_dir)
    df, X_items = load_all_parts(pairs, max_rows=args.max_rows)

    # target
    y = df["cooking_minutes"].to_numpy().astype(float)

    # split
    train_idx, val_idx, test_idx = split_by_order_id(df, train_q=0.80, val_q=0.90)

    df_train, df_val, df_test = df.iloc[train_idx].copy(), df.iloc[val_idx].copy(), df.iloc[test_idx].copy()
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    X_items_train = X_items[train_idx] if X_items is not None else None
    X_items_val = X_items[val_idx] if X_items is not None else None
    X_items_test = X_items[test_idx] if X_items is not None else None

    # preprocess
    pre = build_preprocessor(df_train)

    X_train = make_X(pre, df_train, X_items_train, fit=True)
    X_val = make_X(pre, df_val, X_items_val, fit=False)
    X_test = make_X(pre, df_test, X_items_test, fit=False)

    # optional target transform
    def y_to_train_space(y_arr: np.ndarray) -> np.ndarray:
        return np.log1p(y_arr) if args.use_log_target else y_arr

    def pred_to_minutes(p: np.ndarray) -> np.ndarray:
        if args.use_log_target:
            p = np.expm1(p)
        # safety clip
        return np.clip(p, 0.0, None)

    y_train_t = y_to_train_space(y_train)

    # ------------- Models -------------
    results = []
    preds_test: Dict[str, np.ndarray] = {}

    # Baseline 1: global median
    m1 = GlobalMedianBaseline().fit(y_train)
    pred = m1.predict(len(y_test))
    preds_test["baseline_median"] = pred
    met = compute_metrics(y_test, pred)
    results.append(("baseline_median", met))

    # Baseline 2: point-hour mean
    m2 = PointHourMeanBaseline().fit(df_train, y_train)
    pred = m2.predict(df_test)
    preds_test["baseline_point_hour_mean"] = pred
    met = compute_metrics(y_test, pred)
    results.append(("baseline_point_hour_mean", met))

    # Model 3: Ridge (linear, strong baseline for sparse)
    ridge = Ridge(alpha=args.ridge_alpha, solver="sag", random_state=args.seed)
    ridge.fit(X_train, y_train_t)
    pred = pred_to_minutes(ridge.predict(X_test))
    preds_test["ridge"] = pred
    met = compute_metrics(y_test, pred)
    results.append(("ridge", met))

    # Model 4: SGDRegressor
    sgd = SGDRegressor(
        loss=args.sgd_loss,
        alpha=args.sgd_alpha,
        epsilon=args.sgd_epsilon if args.sgd_loss == "huber" else 0.1,
        penalty="l2",
        learning_rate="invscaling",
        eta0=0.01,
        power_t=0.25,
        max_iter=50,
        tol=1e-4,
        random_state=args.seed,
    )

    sgd.fit(X_train, y_train_t)
    pred = pred_to_minutes(sgd.predict(X_test))

    preds_test["sgd"] = pred
    met = compute_metrics(y_test, pred)
    results.append(("sgd", met))

    # ------------- Save metrics -------------
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

    # ------------- Plots -------------
    plot_model_bars(metrics_df, out_dir / "mae_by_model.png")
    plot_pred_vs_true(y_test, preds_test, out_dir / "pred_vs_true.png", sample_n=20000)
    plot_residual_hist(y_test, preds_test, out_dir)
    plot_calibration_deciles(y_test, preds_test, out_dir)

    # ------------- Save models -------------
    joblib.dump(pre, out_dir / "preprocessor.joblib")
    joblib.dump(ridge, out_dir / "ridge.joblib")
    joblib.dump(sgd, out_dir / "sgd.joblib")

    info = {
        "use_log_target": bool(args.use_log_target),
        "ridge_alpha": args.ridge_alpha,
        "sgd": {
            "alpha": args.sgd_alpha,
            "loss": args.sgd_loss,
            "epsilon": args.sgd_epsilon,
        },
        "notes": "X = [OneHot(cats)+Scaled(nums)] + hashed_items_sparse (if provided). Split by order_id quantiles.",
        "files": {
            "metrics": str(metrics_path),
            "plots_dir": str(out_dir),
        },
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(metrics_df.to_string(index=False))
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
