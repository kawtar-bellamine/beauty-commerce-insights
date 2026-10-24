"""Data-quality checks on the raw and staged data. Each check returns how many rows break a rule."""
from __future__ import annotations

import pandas as pd

# (id, severity, description, SQL returning the number of offending rows, total rows SQL)
CHECKS = [
    ("Q01", "blocking", "Duplicate order ids",
     "SELECT COUNT(*) - COUNT(DISTINCT order_id) FROM raw_orders", "SELECT COUNT(*) FROM raw_orders"),
    ("Q02", "blocking", "Order items pointing to an unknown order",
     "SELECT COUNT(*) FROM raw_items i LEFT JOIN raw_orders o USING (order_id) WHERE o.order_id IS NULL",
     "SELECT COUNT(*) FROM raw_items"),
    ("Q03", "blocking", "Order items pointing to an unknown product",
     "SELECT COUNT(*) FROM raw_items i LEFT JOIN raw_products p USING (product_id) WHERE p.product_id IS NULL",
     "SELECT COUNT(*) FROM raw_items"),
    ("Q04", "warning", "Products without a category",
     "SELECT COUNT(*) FROM raw_products WHERE product_category_name IS NULL", "SELECT COUNT(*) FROM raw_products"),
    ("Q05", "warning", "Categories missing from the English translation table",
     "SELECT COUNT(DISTINCT p.product_category_name) FROM raw_products p LEFT JOIN raw_category_translation t "
     "USING (product_category_name) WHERE p.product_category_name IS NOT NULL AND t.product_category_name IS NULL",
     "SELECT COUNT(DISTINCT product_category_name) FROM raw_products"),
    ("Q06", "warning", "Orders with more than one review (latest kept)",
     "SELECT COUNT(*) FROM (SELECT order_id FROM raw_reviews GROUP BY 1 HAVING COUNT(*) > 1)",
     "SELECT COUNT(DISTINCT order_id) FROM raw_reviews"),
    ("Q07", "warning", "Delivered orders without a delivery date",
     "SELECT COUNT(*) FROM stg_orders WHERE status = 'delivered' AND delivered_at IS NULL",
     "SELECT COUNT(*) FROM stg_orders WHERE status = 'delivered'"),
    ("Q08", "blocking", "Delivered before being purchased",
     "SELECT COUNT(*) FROM stg_orders WHERE delivered_at < purchased_at", "SELECT COUNT(*) FROM stg_orders"),
    ("Q09", "warning", "Handed to the carrier before payment approval",
     "SELECT COUNT(*) FROM stg_orders WHERE shipped_at < approved_at", "SELECT COUNT(*) FROM stg_orders"),
    ("Q10", "warning", "Orders where payments differ from items + freight by more than 1 BRL",
     "SELECT COUNT(*) FROM fct_orders WHERE paid IS NOT NULL AND gmv IS NOT NULL AND ABS(paid - gmv - freight) > 1",
     "SELECT COUNT(*) FROM fct_orders WHERE paid IS NOT NULL AND gmv IS NOT NULL"),
    ("Q11", "warning", "Orders without any payment record",
     "SELECT COUNT(*) FROM fct_orders WHERE paid IS NULL", "SELECT COUNT(*) FROM fct_orders"),
    ("Q12", "blocking", "Items with a non-positive price",
     "SELECT COUNT(*) FROM stg_items WHERE price <= 0", "SELECT COUNT(*) FROM stg_items"),
]


def run_checks(con) -> pd.DataFrame:
    rows = []
    for cid, severity, desc, bad_sql, total_sql in CHECKS:
        bad = int(con.execute(bad_sql).fetchone()[0])
        total = int(con.execute(total_sql).fetchone()[0])
        rows.append({"check": cid, "severity": severity, "description": desc, "rows_failing": bad,
                     "rows_checked": total, "failing_pct": round(100 * bad / total, 3) if total else 0.0,
                     "status": "PASS" if bad == 0 else ("FAIL" if severity == "blocking" else "WARN")})
    return pd.DataFrame(rows)
