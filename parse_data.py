import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.environ["DB_HOST"]
DB_PORT = int(os.environ.get("DB_PORT", "3306"))
DB_NAME = os.environ["DB_NAME"]
DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]

# Куда складываем датасет (лучше папкой из нескольких parquet-файлов, а не одним гигантским файлом)
OUT_DIR = Path("datasets/orders_cooking_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ВАЖНО: укажите реальные имена таблиц (ниже - примерные)
ORDERS_TABLE = "orders_order"
AUDIT_TABLE = "orders_orderaudit"
ITEM_TABLE = "orders_item"

# Фильтр по датам — ОЧЕНЬ рекомендую, чтобы не тянуть 8 лет сразу
DATE_FROM = "2024-01-01"  # стартово возьмите 6-18 месяцев, потом расширите при необходимости

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    pool_pre_ping=True,
    pool_recycle=3600,
)

# 1) Находим kitchen_time на заказ
# 2) Находим cooked_time на заказ (минимальное cooked_time > kitchen_time)
# 3) Добавляем агрегаты по items
# 4) Считаем cooking_minutes прямо в SQL
QUERY = f"""
WITH kitchen AS (
    SELECT
        order_id,
        MIN(updated_at) AS kitchen_time
    FROM {AUDIT_TABLE}
    WHERE status = 'kitchen'
    GROUP BY order_id
),
cooked AS (
    SELECT
        a.order_id,
        MIN(a.updated_at) AS cooked_time
    FROM {AUDIT_TABLE} a
    JOIN kitchen k ON k.order_id = a.order_id
    WHERE a.status = 'cooked'
      AND a.updated_at > k.kitchen_time
    GROUP BY a.order_id
),
items AS (
    SELECT
        order_id,
        COALESCE(SUM(quantity), 0) AS items_qty_total,
        COUNT(*) AS items_lines
    FROM {ITEM_TABLE}
    GROUP BY order_id
)
SELECT
    o.id AS order_id,
    o.point_id,
    o.city_id,
    o.delivery_method,
    o.delivery_at_client,
    o.created_at,
    o.persons,
    i.items_qty_total,
    i.items_lines,
    k.kitchen_time,
    c.cooked_time,
    TIMESTAMPDIFF(MINUTE, k.kitchen_time, c.cooked_time) AS cooking_minutes
FROM {ORDERS_TABLE} o
JOIN kitchen k ON k.order_id = o.id
JOIN cooked  c ON c.order_id = o.id
LEFT JOIN items i ON i.order_id = o.id
WHERE o.created_at >= :date_from
  AND c.cooked_time IS NOT NULL
  AND k.kitchen_time IS NOT NULL
  AND c.cooked_time > k.kitchen_time
"""

def main():
    chunksize = 200_000  # подберите под память
    part = 0

    with engine.connect() as conn:
        result_iter = pd.read_sql_query(
            sql=text(QUERY),
            con=conn,
            params={"date_from": DATE_FROM},
            chunksize=chunksize,
        )

        for chunk in result_iter:
            # Базовая чистка
            chunk = chunk.dropna(subset=["cooking_minutes", "kitchen_time", "cooked_time"])
            chunk = chunk[(chunk["cooking_minutes"] > 0) & (chunk["cooking_minutes"] < 240)]  # пример отсечения выбросов

            # Фичи времени (можно сделать сразу здесь, можно отдельным скриптом)
            # ВАЖНО: если timestamps в БД уже в UTC — явно фиксируйте utc=True
            chunk["kitchen_time"] = pd.to_datetime(chunk["kitchen_time"], utc=True, errors="coerce")
            chunk["created_at"] = pd.to_datetime(chunk["created_at"], utc=True, errors="coerce")

            chunk["weekday"] = chunk["kitchen_time"].dt.weekday
            chunk["hour"] = chunk["kitchen_time"].dt.hour

            out_path = OUT_DIR / f"part-{part:05d}.parquet"
            chunk.to_parquet(out_path, index=False)
            parquet_path = OUT_DIR / f"part-{part:05d}.parquet"
            csv_path = OUT_DIR / f"part-{part:05d}.csv.gz"

            try:
                chunk.to_parquet(parquet_path, index=False)
                print(f"Saved {len(chunk):,} rows -> {parquet_path}")
            except ImportError:
                chunk.to_csv(csv_path, index=False, compression="gzip")
                print(f"Saved {len(chunk):,} rows -> {csv_path} (parquet engine missing)")

            print(f"Saved {len(chunk):,} rows -> {out_path}")
            part += 1

    # Сохраним немного метаданных о датасете
    meta_path = OUT_DIR / "_meta.json"
    meta_path.write_text(
        f'{{"date_from":"{DATE_FROM}","chunksize":{chunksize},"query_version":"v1"}}',
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()
