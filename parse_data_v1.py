from datetime import datetime, timedelta
from pathlib import Path
import os

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.environ["DB_HOST"]
DB_PORT = int(os.environ.get("DB_PORT", "3306"))
DB_NAME = os.environ["DB_NAME"]
DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=True,  # выключите, когда всё заработает
)

OUT_DIR = Path("datasets/orders_cooking_vB_packed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

QUERY = """
WITH orders_in_range AS (
    SELECT id, point_id, city_id, delivery_method, delivery_at_client, created_at, persons
    FROM orders_order
    WHERE created_at >= :start_dt
      AND created_at <  :end_dt
),
kitchen_a AS (
    SELECT
        a.order_id,
        a.updated_at AS kitchen_time,
        a.dump_item  AS kitchen_dump_item,
        ROW_NUMBER() OVER (PARTITION BY a.order_id ORDER BY a.updated_at ASC, a.id ASC) AS rn
    FROM orders_orderaudit a
    JOIN orders_in_range o ON o.id = a.order_id
    WHERE a.status = 'kitchen'
),
kitchen AS (
    SELECT order_id, kitchen_time, kitchen_dump_item
    FROM kitchen_a
    WHERE rn = 1
),
packed_a AS (
    SELECT
        a.order_id,
        a.updated_at AS packed_time,
        ROW_NUMBER() OVER (PARTITION BY a.order_id ORDER BY a.updated_at ASC, a.id ASC) AS rn
    FROM orders_orderaudit a
    JOIN kitchen k ON k.order_id = a.order_id
    WHERE a.status = 'packed'
      AND a.updated_at > k.kitchen_time
),
packed AS (
    SELECT order_id, packed_time
    FROM packed_a
    WHERE rn = 1
)
SELECT
    o.id AS order_id,
    o.point_id,
    o.city_id,
    o.delivery_method,
    o.delivery_at_client,
    o.created_at,
    o.persons,
    k.kitchen_time,
    p.packed_time,
    TIMESTAMPDIFF(MINUTE, k.kitchen_time, p.packed_time) AS cooking_minutes,
    k.kitchen_dump_item AS dump_item
FROM orders_in_range o
JOIN kitchen k ON k.order_id = o.id
JOIN packed  p ON p.order_id = o.id
WHERE p.packed_time > k.kitchen_time;
"""

START = datetime(2025, 6, 1)
END   = datetime(2026, 1, 9)

def daterange(start, end, step_days=7):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=step_days), end)
        yield cur, nxt
        cur = nxt

def main():
    with engine.connect() as conn:
        for start_dt, end_dt in daterange(START, END, step_days=7):
            df = pd.read_sql_query(
                text(QUERY),
                conn,
                params={"start_dt": start_dt, "end_dt": end_dt},
            )

            if df.empty:
                print(f"{start_dt}..{end_dt}: empty")
                continue

            # минимальная чистка (настройте порог позже)
            df = df.dropna(subset=["cooking_minutes", "dump_item", "kitchen_time", "packed_time"])
            df = df[(df["cooking_minutes"] > 0) & (df["cooking_minutes"] < 600)]

            out_path = OUT_DIR / f"start={start_dt.date()}_end={end_dt.date()}.parquet"
            df.to_parquet(out_path, index=False)
            print(f"{start_dt}..{end_dt}: saved {len(df):,} -> {out_path}")

if __name__ == "__main__":
    main()
