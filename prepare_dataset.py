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
# dump_item parsing
# ----------------------------

# trailing " x 3" / "x3"
_QTY_RE = re.compile(r"\s*x\s*(\d+)\s*$", flags=re.IGNORECASE)

_HAS_LATIN = re.compile(r"[A-Za-z]")
_HAS_CYR = re.compile(r"[А-Яа-яЁё]")

# единицы/размеры, которые не должны становиться отдельной "позицией"
_SIZE_ONLY_RE = re.compile(
    r"^\s*\d+(?:[.,]\d+)?\s*(см|мм|м|кг|г|гр|л|мл|ml|l|kg)\.?\s*$",
    flags=re.IGNORECASE,
)

# иногда встречаются "25см" без пробела
_SIZE_ONLY_RE2 = re.compile(
    r"^\s*\d+(?:[.,]\d+)?\s*(см|мм|м|кг|г|гр|л|мл)\s*$",
    flags=re.IGNORECASE,
)

# если совсем нет букв (только цифры/символы) — мусор
_HAS_ANY_LETTERS = re.compile(r"[A-Za-zА-Яа-яЁё]")


def _split_qty(name: str) -> Tuple[str, Optional[int]]:
    name = name.replace("\n", " ").strip()
    m = _QTY_RE.search(name)
    if not m:
        return name, None
    qty = int(m.group(1))
    base = name[: m.start()].strip()
    return base, qty


def _normalize_key(s: str) -> str:
    s = s.strip().lower()
    s = " ".join(s.split())  # схлопнуть пробелы
    # убрать частый мусор по краям
    s = s.strip(" -–—:;,.()[]{}\"'")
    return s


def _choose_item_key(base: str) -> Optional[str]:
    """
    Возвращает "каноническое имя позиции" (EN если есть, иначе RU),
    с учетом того, что после ':' может быть размер/модификатор.
    """
    base = base.strip()

    # 1) RU - EN
    left, sep, right = base.rpartition(" - ")
    if sep and right and _HAS_LATIN.search(right):
        key = _normalize_key(right)
        if key and _HAS_ANY_LETTERS.search(key) and not _SIZE_ONLY_RE.match(key) and not _SIZE_ONLY_RE2.match(key):
            return key

    # 2) RU: EN  (если справа латиница)
    left, sep, right = base.rpartition(":")
    if sep:
        right_str = right.strip()
        left_str = left.strip()

        if right_str and _HAS_LATIN.search(right_str):
            key = _normalize_key(right_str)
            if key and _HAS_ANY_LETTERS.search(key) and not _SIZE_ONLY_RE.match(key) and not _SIZE_ONLY_RE2.match(key):
                return key

        # 3) RU: <модификатор>  => берём RU (левую часть), если там есть буквы
        if left_str and _HAS_ANY_LETTERS.search(left_str):
            key = _normalize_key(left_str)
            if key and not _SIZE_ONLY_RE.match(key) and not _SIZE_ONLY_RE2.match(key):
                return key

    # 4) Иначе: берём всё как RU
    if _HAS_ANY_LETTERS.search(base):
        key = _normalize_key(base)
        if key and not _SIZE_ONLY_RE.match(key) and not _SIZE_ONLY_RE2.match(key):
            return key

    return None


def parse_dump_item(
    dump_item: Any,
) -> Tuple[int, int, Dict[str, float]]:
    """
    Returns:
      items_qty_total_all: сумма qty по всем валидным позициям
      items_lines_all: количество валидных позиций (строк) в заказе
      items_dict: {item_key: qty}  (ключ = EN если есть, иначе RU)
    """
    if dump_item is None:
        return 0, 0, {}

    if isinstance(dump_item, bytes):
        try:
            dump_item = dump_item.decode("utf-8", errors="ignore")
        except Exception:
            return 0, 0, {}

    if not isinstance(dump_item, str) or not dump_item:
        return 0, 0, {}

    try:
        arr = orjson.loads(dump_item)
    except Exception:
        return 0, 0, {}

    if not isinstance(arr, list):
        return 0, 0, {}

    total_all = 0
    lines_all = 0
    out: Dict[str, float] = {}

    for obj in arr:
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        if not isinstance(name, str) or not name:
            continue

        base, qty = _split_qty(name)
        if qty is None:
            continue

        key = _choose_item_key(base)
        if not key:
            continue

        lines_all += 1
        total_all += qty
        out[key] = out.get(key, 0.0) + float(qty)

    return total_all, lines_all, out


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
    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        yield batch.to_pandas()


def ensure_dirs(cfg: PrepConfig) -> Tuple[Path, Path]:
    base_dir = cfg.out_dir / "prepared_base"
    base_dir.mkdir(parents=True, exist_ok=True)

    items_dir = cfg.out_dir / "prepared_items_hash"
    items_dir.mkdir(parents=True, exist_ok=True)

    return base_dir, items_dir


def prepare_batch(df: pd.DataFrame, cfg: PrepConfig) -> Tuple[pd.DataFrame, Optional[sparse.csr_matrix]]:
    df = df.dropna(subset=["cooking_minutes", "kitchen_time", "packed_time", "dump_item"])
    df = df[(df["cooking_minutes"] > 0) & (df["cooking_minutes"] < cfg.max_cooking_minutes)]
    if df.empty:
        return df, None

    # datetime
    df["kitchen_time"] = pd.to_datetime(df["kitchen_time"], errors="coerce")
    df["created_at"] = pd.to_datetime(df.get("created_at"), errors="coerce")

    # delivery_at_client: 0/1
    if "delivery_at_client" in df.columns:
        df["delivery_at_client"] = pd.to_numeric(df["delivery_at_client"], errors="coerce").fillna(0).astype("int8")

        # EXCLUDE PREORDERS
        df = df[df["delivery_at_client"] == 0]
        if df.empty:
            return df, None

    df = df.reset_index(drop=True)

    # time features (используем kitchen_time, иначе created_at)
    dt = df["kitchen_time"].where(~df["kitchen_time"].isna(), df["created_at"])
    df["weekday"] = dt.dt.weekday.astype("int8")
    df["hour"] = dt.dt.hour.astype("int8")
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype("int8")

    # parse dump_item -> dict of items
    item_dicts: list[Dict[str, float]] = []
    totals_all = []
    lines_all = []
    unique_all = []

    for s in df["dump_item"].tolist():
        t_all, l_all, d = parse_dump_item(s)
        totals_all.append(t_all)
        lines_all.append(l_all)
        unique_all.append(len(d))
        item_dicts.append(d)

    df["items_qty_total_all"] = pd.Series(totals_all, index=df.index, dtype="int32")
    df["items_lines_all"] = pd.Series(lines_all, index=df.index, dtype="int32")
    df["items_unique_all"] = pd.Series(unique_all, index=df.index, dtype="int32")

    X_items = None
    if cfg.hash_items:
        # Состав заказа как основной признак: hashed bag-of-items
        hasher = FeatureHasher(
            n_features=cfg.hash_dim,
            input_type="dict",
            alternate_sign=False,
        )
        X_items = hasher.transform(item_dicts).tocsr()

    # base features + target
    keep_cols = [
        "order_id",
        "cooking_minutes",
        "point_id",
        "city_id",
        "delivery_method",
        "delivery_at_client",
        "weekday",
        "hour",
        "is_weekend",
        "items_qty_total_all",
        "items_lines_all",
        "items_unique_all",
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

    parser.add_argument("--hash-items", action="store_true", help="Build hashed sparse features from items")
    parser.add_argument("--hash-dim", type=int, default=32768, help="Hashed feature dimension (power of 2 recommended)")

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

    columns = [
        "order_id",
        "point_id",
        "city_id",
        "delivery_method",
        "delivery_at_client",
        "created_at",
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
        "notes": (
            "item_key = EN if present (RU - EN or RU: EN with latin), else RU name; "
            "RU: <size/modifier> -> RU used; size-only keys dropped; keys normalized lower()+spaces."
        ),
    }
    (cfg.out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. total_rows={total_rows:,}. meta.json -> {cfg.out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
