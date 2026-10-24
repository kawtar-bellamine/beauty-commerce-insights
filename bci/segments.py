"""Shopper segmentation of beauty customers.

Only about 2% of beauty customers order twice, so a classic RFM segmentation would put almost everybody in the same
"one-time buyer" box. Segments are therefore built on how people buy: basket value, unit price, number of items,
freight burden, payment in instalments and perfume share. The number of segments is picked by silhouette score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .experience import REGION

FEATURES = ["log_spend", "log_unit_price", "items_per_order", "freight_ratio", "installments", "perfume_share"]


def customers(con) -> pd.DataFrame:
    df = con.execute("""
        SELECT customer_unique_id, ANY_VALUE(customer_state) AS state, COUNT(*) AS orders,
               SUM(gmv) AS spend, SUM(n_items) AS items, SUM(freight) AS freight,
               AVG(installments) AS installments, SUM(perfume_gmv) / NULLIF(SUM(gmv), 0) AS perfume_share,
               AVG(review_score) AS avg_score, AVG(CAST(is_late AS DOUBLE)) AS late_rate
        FROM fct_orders WHERE has_beauty AND status NOT IN ('canceled', 'unavailable') AND gmv IS NOT NULL
        GROUP BY customer_unique_id
        ORDER BY customer_unique_id
    """).df()
    df["log_spend"] = np.log1p(df["spend"])
    df["log_unit_price"] = np.log1p(df["spend"] / df["items"])
    df["items_per_order"] = df["items"] / df["orders"]
    df["freight_ratio"] = df["freight"] / df["spend"].clip(lower=1)
    df["installments"] = df["installments"].fillna(1)
    df["perfume_share"] = df["perfume_share"].fillna(0)
    df["region"] = df["state"].map(REGION).fillna("Other")
    return df


def choose_k(X: np.ndarray, ks=range(3, 7), seed: int = 0, sample: int = 6000) -> tuple[int, dict]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    scores = {}
    for k in ks:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        scores[k] = round(float(silhouette_score(X[idx], labels[idx])), 3)
    return max(scores, key=scores.get), scores


def name_segment(c: pd.Series, overall: pd.Series) -> str:
    """Readable persona name from the segment profile compared with all beauty customers."""
    if c["perfume_share"] > 0.6:
        return "Fragrance buyers"
    if c["items_per_order"] > 1.5 * overall["items_per_order"]:
        return "Routine stockers"
    if c["unit_price"] > 1.6 * overall["unit_price"]:
        return "Premium seekers"
    if c["freight_ratio"] > 1.5 * overall["freight_ratio"]:
        return "Freight-sensitive small baskets"
    if c["installments"] > 1.4 * overall["installments"]:
        return "Instalment planners"
    return "Everyday essentials"


def segment(con, seed: int = 0) -> dict:
    df = customers(con)
    X = StandardScaler().fit_transform(df[FEATURES])
    k, sil = choose_k(X, seed=seed)
    df["segment_id"] = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    df["unit_price"] = df["spend"] / df["items"]
    overall = df[["items_per_order", "unit_price", "freight_ratio", "installments"]].median()
    prof = df.groupby("segment_id").agg(
        customers=("customer_unique_id", "size"), spend_share=("spend", "sum"), avg_spend=("spend", "mean"),
        unit_price=("unit_price", "median"), items_per_order=("items_per_order", "mean"),
        freight_ratio=("freight_ratio", "median"), installments=("installments", "mean"),
        perfume_share=("perfume_share", "mean"), repeat_rate=("orders", lambda s: (s > 1).mean()),
        avg_score=("avg_score", "mean"), late_rate=("late_rate", "mean"))
    prof["spend_share"] = 100 * prof["spend_share"] / prof["spend_share"].sum()
    prof["customer_share"] = 100 * prof["customers"] / prof["customers"].sum()
    names = {}
    for sid, row in prof.iterrows():
        n = name_segment(row, overall)
        while n in names.values():
            n += " (2)"
        names[sid] = n
    prof["segment"] = prof.index.map(names)
    df["segment"] = df["segment_id"].map(names)
    prof = prof.reset_index().sort_values("spend_share", ascending=False)
    cols = ["segment", "customers", "customer_share", "spend_share", "avg_spend", "unit_price", "items_per_order",
            "freight_ratio", "installments", "perfume_share", "repeat_rate", "avg_score", "late_rate"]
    return {"k": k, "silhouette": sil, "profile": prof[cols].round(3), "customers": df,
            "repeat_rate_pct": round(100 * (df["orders"] > 1).mean(), 2)}
