"""Builds data/sample/: a reproducible extract of the real data (3,500 beauty orders + 1,500 others, every
related row kept) so that tests and CI run in seconds without the 64 MB download."""
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bci.data import RAW, SAMPLE, TABLES, load_raw  # noqa: E402

con = duckdb.connect()
load_raw(con, RAW)
con.execute("""
CREATE TABLE pick AS
WITH tagged AS (
  SELECT DISTINCT i.order_id, p.product_category_name IN ('beleza_saude','perfumaria') AS beauty
  FROM raw_items i JOIN raw_products p USING (product_id))
SELECT order_id FROM (SELECT order_id FROM tagged WHERE beauty ORDER BY hash(order_id) LIMIT 3500)
UNION SELECT order_id FROM (SELECT order_id FROM tagged WHERE NOT beauty ORDER BY hash(order_id) LIMIT 1500)
""")
SAMPLE.mkdir(parents=True, exist_ok=True)
q = {
    "raw_orders": "SELECT * FROM raw_orders WHERE order_id IN (SELECT order_id FROM pick)",
    "raw_items": "SELECT * FROM raw_items WHERE order_id IN (SELECT order_id FROM pick)",
    "raw_reviews": "SELECT * FROM raw_reviews WHERE order_id IN (SELECT order_id FROM pick)",
    "raw_payments": "SELECT * FROM raw_payments WHERE order_id IN (SELECT order_id FROM pick)",
    "raw_products": "SELECT * FROM raw_products WHERE product_id IN (SELECT product_id FROM raw_items WHERE order_id IN (SELECT order_id FROM pick))",
    "raw_sellers": "SELECT * FROM raw_sellers WHERE seller_id IN (SELECT seller_id FROM raw_items WHERE order_id IN (SELECT order_id FROM pick))",
    "raw_customers": "SELECT * FROM raw_customers WHERE customer_id IN (SELECT customer_id FROM raw_orders WHERE order_id IN (SELECT order_id FROM pick))",
    "raw_category_translation": "SELECT * FROM raw_category_translation",
}
for t, sql in q.items():
    con.execute(f"COPY ({sql}) TO '{(SAMPLE / TABLES[t]).as_posix()}' (HEADER, DELIMITER ',')")
print(sorted((f.name, f.stat().st_size // 1024) for f in SAMPLE.iterdir()))
