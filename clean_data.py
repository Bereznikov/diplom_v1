from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


DATASET_PATH = Path("datasets/orders_cooking_v1")  # папка с part-xxxxx.parquet
OUT_PATH = Path("datasets/orders_cooking_v1_features.parquet")


def load_parquet_dataset(path: Path) -> pd.DataFrame:
    """
    Читает либо один parquet-файл, либо директорию parquet-partition.
    """
    try:
        return pd.read_parquet(path)
    except Exception:
        # fallback, если в окружении проблемы с pandas.read_parquet
        import pyarrow.parquet as pq
        table = pq.read_table(str(path))
        return table.to_pandas()


def trim_outliers_iqr(s: pd.Series, k: float = 3.0) -> tuple[float, float]:
    """
    Возвращает границы для отсечения выбросов по IQR.
    k=3.0 обычно аккуратно срезает экстремальные хвосты.
    """
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return float(lo), float(hi)


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # --- 1) Приводим типы и базовая валидация ---
    required_cols = {
        "order_id", "point_id", "delivery_method", "delivery_at_client",
        "persons", "items_qty_total", "items_lines",
        "kitchen_time", "packed_time", "cooking_minutes",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")

    # datetime
    df["kitchen_time"] = pd.to_datetime(df["kitchen_time"], errors="coerce")
    df["packed_time"] = pd.to_datetime(df["packed_time"], errors="coerce")

    # bool
    # в вашем датасете delivery_at_client = 0/1 (int)
    df["delivery_at_client"] = df["delivery_at_client"].astype("int64").astype("bool")

    # числовые
    df["items_qty_total"] = pd.to_numeric(df["items_qty_total"], errors="coerce")
    df["items_lines"] = pd.to_numeric(df["items_lines"], errors="coerce")
    df["cooking_minutes"] = pd.to_numeric(df["cooking_minutes"], errors="coerce")

    # убираем явный мусор
    df = df.dropna(subset=["kitchen_time", "packed_time", "cooking_minutes", "point_id", "delivery_method"])
    df = df[df["cooking_minutes"] > 0]

    # (опционально) дедуп по заказу: если вдруг попались дубли, оставляем самый поздний packed_time
    df = df.sort_values(["order_id", "packed_time"]).drop_duplicates("order_id", keep="last")

    # --- 2) Отсечение выбросов по таргету (лучше, чем жесткое <240 на старте) ---
    lo, hi = trim_outliers_iqr(df["cooking_minutes"], k=3.0)
    lo = max(lo, 1.0)
    df = df[(df["cooking_minutes"] >= lo) & (df["cooking_minutes"] <= hi)]

    # --- 3) Фичи времени (по фактическому времени начала кухни) ---
    # Даже если у вас уже есть weekday/hour — пересчитываем, чтобы гарантировать согласованность
    df["kitchen_weekday"] = df["kitchen_time"].dt.weekday.astype("int8")  # 0..6
    df["kitchen_hour"] = df["kitchen_time"].dt.hour.astype("int8")        # 0..23
    df["is_weekend"] = (df["kitchen_weekday"] >= 5).astype("int8")
    df["kitchen_month"] = df["kitchen_time"].dt.month.astype("int8")

    # циклическое кодирование времени (часто помогает моделям)
    df["hour_sin"] = np.sin(2 * np.pi * df["kitchen_hour"] / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * df["kitchen_hour"] / 24).astype("float32")
    df["weekday_sin"] = np.sin(2 * np.pi * df["kitchen_weekday"] / 7).astype("float32")
    df["weekday_cos"] = np.cos(2 * np.pi * df["kitchen_weekday"] / 7).astype("float32")

    # --- 4) Фичи “размера заказа” ---
    # items_qty_total у вас float, но это “сумма quantity” — можно оставить float32
    df["items_qty_total"] = df["items_qty_total"].fillna(0).clip(lower=0).astype("float32")
    df["items_lines"] = df["items_lines"].fillna(0).clip(lower=0).astype("int32")

    df["items_per_person"] = (df["items_qty_total"] / df["persons"]).astype("float32")
    df["log_items_qty_total"] = np.log1p(df["items_qty_total"]).astype("float32")

    # --- 5) Категориальные (для one-hot/катбуст) ---
    # Для sklearn OneHotEncoder удобно держать как string
    df["point_id"] = df["point_id"].astype("int64").astype("string")
    df["delivery_method"] = df["delivery_method"].astype("string")

    # --- 6) Финальный набор ---
    # Сохраним только то, что нужно для обучения/аналитики
    feature_cols = [
        # категориальные
        "point_id", "delivery_method",
        # бинарные / числовые
        "delivery_at_client",
        "persons", "items_qty_total", "items_lines", "items_per_person", "log_items_qty_total",
        # время
        "kitchen_weekday", "kitchen_hour", "is_weekend", "kitchen_month",
        "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
        # опционально: "city_id" (если реально несёт смысл)
        # "city_id",
    ]

    out = df[["order_id", "kitchen_time", "packed_time", "cooking_minutes"] + feature_cols].copy()
    out["cooking_minutes"] = out["cooking_minutes"].astype("float32")

    return out


def main():
    df_raw = load_parquet_dataset(DATASET_PATH)
    df_feat = build_features(df_raw)

    print("Raw:", df_raw.shape)
    print("Features:", df_feat.shape)
    print(df_feat.head(3).to_string(index=False))

    df_feat.to_parquet(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
