"""Runs the whole analysis once and returns every output (used by the study script, the dashboard and the tests)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import experience as E
from . import forecast as F
from . import segments as S
from .quality import run_checks
from .warehouse import build


def kpis(con) -> dict:
    r = con.execute("""
        SELECT COUNT(*) FILTER (WHERE has_beauty),
               SUM(beauty_gmv) FILTER (WHERE status NOT IN ('canceled','unavailable')),
               COUNT(DISTINCT customer_unique_id) FILTER (WHERE has_beauty),
               AVG(review_score) FILTER (WHERE has_beauty),
               AVG(CAST(is_late AS DOUBLE)) FILTER (WHERE has_beauty AND status = 'delivered'),
               SUM(gmv) FILTER (WHERE status NOT IN ('canceled','unavailable'))
        FROM fct_orders
    """).fetchone()
    return {"beauty_orders": int(r[0]), "beauty_gmv_brl": round(float(r[1]), 0), "beauty_customers": int(r[2]),
            "beauty_avg_review": round(float(r[3]), 2), "beauty_late_pct": round(100 * float(r[4]), 1),
            "beauty_share_of_marketplace_gmv_pct": round(100 * float(r[1]) / float(r[5]), 1)}


def monthly(con) -> pd.DataFrame:
    return con.execute("""
        SELECT DATE_TRUNC('month', purchased_at) AS month,
               SUM(beauty_gmv - perfume_gmv) AS health_beauty, SUM(perfume_gmv) AS perfumery,
               COUNT(*) AS orders, AVG(review_score) AS avg_review
        FROM fct_orders WHERE has_beauty AND status NOT IN ('canceled','unavailable')
          AND purchased_at >= '2017-01-01' AND purchased_at < '2018-09-01'
        GROUP BY 1 ORDER BY 1
    """).df()


def by_state(con) -> pd.DataFrame:
    return con.execute("""
        SELECT customer_state AS state, COUNT(*) AS orders, SUM(beauty_gmv) AS gmv,
               AVG(delivery_days) AS delivery_days, AVG(CAST(is_late AS DOUBLE)) * 100 AS late_pct,
               AVG(review_score) AS avg_review
        FROM fct_orders WHERE has_beauty AND status = 'delivered'
        GROUP BY 1 ORDER BY gmv DESC
    """).df()


def run(db_path: str | Path = ":memory:", sample: bool = False, folder=None, n_origins: int = 8) -> dict:
    con = build(db_path, sample=sample, folder=folder)
    out = {"con": con, "quality": run_checks(con), "kpis": kpis(con), "monthly": monthly(con), "states": by_state(con)}
    series = F.complete_series(con.execute("SELECT * FROM daily_beauty_sales").df())
    bt = F.backtest(series, n_origins=n_origins)
    sc = F.score(bt)
    best = sc.iloc[0]["model"]
    out.update(series=series, backtest=bt, forecast_scores=sc, best_forecast_model=best,
               forecast=F.forecast_ahead(series, best, bt))
    orders = E.beauty_orders(con)
    exp = E.train_and_test(orders)
    out.update(orders=orders, delay_effect=E.delay_effect(orders), risk=exp, themes=E.complaint_themes(orders))
    out["segments"] = S.segment(con)
    return out
