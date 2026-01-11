from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import orjson
import pandas as pd
import pyarrow.dataset as ds
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher


# ----------------------------
# dump_item parsing (fast)
# ----------------------------

# trailing " x 3"
_QTY_RE = re.compile(r"\s*x\s*(\d+)\s*$", flags=re.IGNORECASE)

# latin fragments fallback
_LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9 _\-\(\)'.&/]+")
_HAS_LATIN = re.compile(r"[A-Za-z]")


def _split_qty(name: str) -> Tuple[str, Optional[int]]:
    name = name.replace("\n", " ").strip()
    m = _QTY_RE.search(name)
    if not m:
        return name, None
    qty = int(m.group(1))
    base = name[: m.start()].strip()
    return base, qty


def _extract_en(base: str) -> Optional[str]:
    """
    Supported formats:
      - "RU - EN"
      - "RU: EN"
    Accept EN only if it contains latin letters.
    Fallback: last latin fragment in whole base.
    """
    base = base.strip()

    # RU - EN
    left, sep, right = base.rpartition(" - ")
    if sep and right and _HAS_LATIN.search(right):
        return right.strip()

    # RU: EN (but RU: 25см -> no latin -> ignore)
    left, sep, right = base.rpartition(":")
    if sep and right and _HAS_LATIN.search(right):
        return right.strip()

    # Fallback: take last latin fragment
    matches = _LATIN_RE.findall(base)
    if matches:
        en = matches[-1].strip()
        if _HAS_LATIN.search(en):
            return en

    return None


def parse_dump_item_stats(
    dump_item: Any,
    *,
    need_en_dict: bool,
) -> Tuple[int, int, int, int, Optional[Dict[str, float]]]:
    """
    Returns:
      items_qty_total_all: int
      items_lines_all: int  (count of entries with parsed qty)
      items_qty_total_en: int
      items_unique_en: int
      en_dict (optional): {en_name_normalized: qty}
    """
    if dump_item is None:
        return 0, 0, 0, 0, {} if need_en_dict else None

    if isinstance(dump_item, bytes):
        try:
            dump_item = dump_item.decode("utf-8", errors="ignore")
        except Exception:
            return 0, 0, 0, 0, {} if need_en_dict else None

    if not isinstance(dump_item, str) or not dump_item:
        return 0, 0, 0, 0, {} if need_en_dict else None

    try:
        arr = orjson.loads(dump_item)
    except Exception:
        return 0, 0, 0, 0, {} if need_en_dict else None

    if not isinstance(arr, list):
        return 0, 0, 0, 0, {} if need_en_dict else None

    total_all = 0
    lines_all = 0

    en_dict: Optional[Dict[str, float]] = {} if need_en_dict else None

    for obj in arr:
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        if not isinstance(name, str) or not name:
            continue

        base, qty = _split_qty(name)
        if qty is None:
            continue

        lines_all += 1
        total_all += qty

        en = _extract_en(base)
        if not en:
            continue

        # normalize to reduce sparsity (NEW/new -> new)
        en_norm = " ".join(en.split()).lower()

        if en_dict is not None:
            en_dict[en_norm] = en_dict.get(en_norm, 0.0) + float(qty)

    if en_dict is None:
        return total_all, lines_all, 0, 0, None

    total_en = int(sum(en_dict.values()))
    unique_en = len(en_dict)
    return total_all, lines_all, total_en, unique_en, en_dict


# ----------------------------
# Batch processing
# ----------------------------

@dataclass(frozen=True)
class PrepConfig:
    input_dir: Path
    out_dir: Path
    batch_size: int
    max_cooking_minutes: int
    hash_items: bool
    hash_dim: int


def iter_parquet_batches(input_dir: Path, columns: list[str], batch_size: int) -> Iterable[pd.DataFrame]:
    dataset = ds.dataset(str(input_dir), format="parquet")
    scanner = dataset.scan(columns=columns, batch_size=batch_size)
    for batch in scanner.to_batches():
        yield batch.to_pandas()


def ensure_dirs(cfg: PrepConfig) -> Tuple[Path, Path]:
    base_dir = cfg.out_dir / "prepared_base"
    base_dir.mkdir(parents=True, exist_ok=True)

    items_dir = cfg.out_dir / "prepared_items_hash"
    items_dir.mkdir(parents=True, exist_ok=True)

    return base_dir, items_dir


def prepare_batch(df: pd.DataFrame, cfg: PrepConfig) -> Tuple[pd.DataFrame, Optional[sparse.csr_matrix]]:
    # target cleaning
    df = df.dropna(subset=["cooking_minutes", "kitchen_time", "packed_time", "dump_item"])
    df = df[(df["cooking_minutes"] > 0) & (df["cooking_minutes"] < cfg.max_cooking_minutes)]
    if df.empty:
        return df, None

    # types
    df["kitchen_time"] = pd.to_datetime(df["kitchen_time"], errors="coerce")
    df["created_at"] = pd.to_datetime(df.get("created_at"), errors="coerce")

    # delivery_at_client is 0/1 int in your parquet
    if "delivery_at_client" in df.columns:
        df["delivery_at_client"] = df["delivery_at_client"].fillna(0).astype("int8")

    # persons can be 0 in your sample; normalize to at least 1 (optional but usually sane)
    if "persons" in df.columns:
        df["persons"] = pd.to_numeric(df["persons"], errors="coerce").fillna(1).astype("int32")
        df.loc[df["persons"] < 1, "persons"] = 1

    # time features: use kitchen_time; fallback to created_at if needed
    dt = df["kitchen_time"].where(~df["kitchen_time"].isna(), df["created_at"])
    df["weekday"] = dt.dt.weekday.astype("int8")
    df["hour"] = dt.dt.hour.astype("int8")
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype("int8")

    # parse dump_item
    need_en_dict = cfg.hash_items
    totals_all = []
    lines_all = []
    totals_en = []
    unique_en = []
    en_dicts = [] if need_en_dict else None

    for s in df["dump_item"].tolist():
        t_all, l_all, t_en, u_en, d_en = parse_dump_item_stats(s, need_en_dict=need_en_dict)
        totals_all.append(t_all)
        lines_all.append(l_all)
        totals_en.append(t_en)
        unique_en.append(u_en)
        if need_en_dict and en_dicts is not None:
            en_dicts.append(d_en or {})

    df["items_qty_total_all"] = pd.Series(totals_all, dtype="int32")
    df["items_lines_all"] = pd.Series(lines_all, dtype="int32")
    df["items_qty_total_en"] = pd.Series(totals_en, dtype="int32")
    df["items_unique_en"] = pd.Series(unique_en, dtype="int32")

    X_items = None
    if cfg.hash_items and en_dicts is not None:
        hasher = FeatureHasher(
            n_features=cfg.hash_dim,
            input_type="dict",
            alternate_sign=False,  # стабильнее для интерпретации
        )
        X_items = hasher.transform(en_dicts).tocsr()

    # keep only base features (+ target)
    keep_cols = [
        "order_id",
        "cooking_minutes",
        "point_id",
        "city_id",
        "delivery_method",
        "delivery_at_client",
        "persons",
        "weekday",
        "hour",
        "is_weekend",
        "items_qty_total_all",
        "items_lines_all",
        "items_qty_total_en",
        "items_unique_en",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df_out = df[keep_cols].copy()

    return df_out, X_items


def save_sparse_matrix(path: Path, X: sparse.csr_matrix) -> None:
    sparse.save_npz(str(path), X)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory with raw parquet parts")
    parser.add_argument("--output", required=True, help="Output directory for prepared dataset")
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--max-cooking-minutes", type=int, default=600)

    parser.add_argument("--hash-items", action="store_true", help="Build hashed sparse features from EN items")
    parser.add_argument("--hash-dim", type=int, default=4096, help="Hashed feature dimension (power of 2 recommended)")

    args = parser.parse_args()

    cfg = PrepConfig(
        input_dir=Path(args.input),
        out_dir=Path(args.output),
        batch_size=args.batch_size,
        max_cooking_minutes=args.max_cooking_minutes,
        hash_items=args.hash_items,
        hash_dim=args.hash_dim,
    )

    base_dir, items_dir = ensure_dirs(cfg)

    # match your real parquet schema
    columns = [
        "order_id",
        "point_id",
        "city_id",
        "delivery_method",
        "delivery_at_client",
        "created_at",
        "persons",
        "kitchen_time",
        "packed_time",
        "cooking_minutes",
        "dump_item",
    ]

    part = 0
    total_rows = 0

    for df_batch in iter_parquet_batches(cfg.input_dir, columns=columns, batch_size=cfg.batch_size):
        df_out, X_items = prepare_batch(df_batch, cfg)
        if df_out.empty:
            continue

        out_base = base_dir / f"part-{part:06d}.parquet"
        df_out.to_parquet(out_base, index=False)

        if cfg.hash_items and X_items is not None:
            out_items = items_dir / f"items-part-{part:06d}.npz"
            save_sparse_matrix(out_items, X_items)

        total_rows += len(df_out)
        print(f"part={part:06d} rows={len(df_out):,} -> {out_base.name}")
        part += 1

    meta = {
        "input_dir": str(cfg.input_dir),
        "rows": total_rows,
        "batch_size": cfg.batch_size,
        "max_cooking_minutes": cfg.max_cooking_minutes,
        "hash_items": cfg.hash_items,
        "hash_dim": cfg.hash_dim if cfg.hash_items else None,
        "base_dir": str(base_dir),
        "items_dir": str(items_dir) if cfg.hash_items else None,
        "notes": "EN items extracted from 'RU - EN' or 'RU: EN' if EN contains latin letters; normalized to lower().",
    }
    (cfg.out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. total_rows={total_rows:,}. meta.json -> {cfg.out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
