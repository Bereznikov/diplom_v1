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

OUT_DIR = Path("datasets/orders_cooking_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATE_FROM = "2024 -01-01"  # для проверки поставь пораньше; потом вернёшь нужный период

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=True,  # включи, если хочешь видеть SQL в логах
)

QUERY = """
WITH kitchen AS (
    SELECT
        order_id,
        MIN(updated_at) AS kitchen_time
    FROM orders_orderaudit
    WHERE status = 'kitchen'
    GROUP BY order_id
),
packed AS (
    SELECT
        a.order_id,
        MIN(a.updated_at) AS packed_time
    FROM orders_orderaudit a
    JOIN kitchen k ON k.order_id = a.order_id
    WHERE a.status = 'packed'
      AND a.updated_at > k.kitchen_time
    GROUP BY a.order_id
),
items AS (
    SELECT
        order_id,
        COALESCE(SUM(quantity), 0) AS items_qty_total,
        COUNT(*) AS items_lines
    FROM orders_item
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
    p.packed_time,
    TIMESTAMPDIFF(MINUTE, k.kitchen_time, p.packed_time) AS cooking_minutes
FROM orders_order o
JOIN kitchen k ON k.order_id = o.id
JOIN packed  p ON p.order_id = o.id
LEFT JOIN items i ON i.order_id = o.id
WHERE o.created_at >= :date_from
  AND p.packed_time > k.kitchen_time
"""

def main():
    chunksize = 200_000
    part = 0

    with engine.connect() as conn:
        # Быстрые sanity checks (по желанию)
        db = pd.read_sql(text("SELECT DATABASE() AS db"), conn)
        print(db.to_string(index=False))

        iter_df = pd.read_sql_query(
            sql=text(QUERY),
            con=conn,
            params={"date_from": DATE_FROM},
            chunksize=chunksize,
        )

        for chunk in iter_df:
            print(f"raw chunk rows: {len(chunk):,}")
            if chunk.empty:
                print("chunk is empty -> skip")
                continue

            # Преобразуем даты
            chunk["kitchen_time"] = pd.to_datetime(chunk["kitchen_time"], errors="coerce")
            chunk["packed_time"] = pd.to_datetime(chunk["packed_time"], errors="coerce")
            chunk["created_at"] = pd.to_datetime(chunk["created_at"], errors="coerce")

            # Базовая чистка
            chunk = chunk.dropna(subset=["kitchen_time", "packed_time", "cooking_minutes"])

            # Фильтры по здравому смыслу (временно можно ослабить для дебага)
            chunk = chunk[chunk["cooking_minutes"] > 0]
            chunk = chunk[chunk["cooking_minutes"] < 240]  # 4 часа; подстрой под бизнес

            if chunk.empty:
                print("all rows filtered out -> skip")
                continue

            # Фичи времени (по кухне)
            chunk["weekday"] = chunk["kitchen_time"].dt.weekday
            chunk["hour"] = chunk["kitchen_time"].dt.hour

            out_path = OUT_DIR / f"part-{part:05d}.parquet"
            chunk.to_parquet(out_path, index=False)
            print(f"Saved {len(chunk):,} rows -> {out_path}")

            part += 1

    print("Done.")

if __name__ == "__main__":
    main()
