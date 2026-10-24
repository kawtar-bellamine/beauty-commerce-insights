"""Customer experience on beauty orders: what drives a bad review (1 or 2 stars), and which orders to call first.

The model scores each order the day it is delivered, using only what is known at that moment
(promised and actual delivery dates, freight, basket, seller location, payment). It is trained on the past and
tested on later months, so the score reflects how it would perform in use.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REGION = {"SP": "Southeast", "RJ": "Southeast", "MG": "Southeast", "ES": "Southeast", "PR": "South", "SC": "South",
          "RS": "South", "DF": "Centre-West", "GO": "Centre-West", "MT": "Centre-West", "MS": "Centre-West",
          "BA": "Northeast", "PE": "Northeast", "CE": "Northeast", "MA": "Northeast", "PB": "Northeast",
          "RN": "Northeast", "AL": "Northeast", "SE": "Northeast", "PI": "Northeast", "PA": "North", "AM": "North",
          "TO": "North", "RO": "North", "AC": "North", "AP": "North", "RR": "North"}

NUMERIC = ["delay_days", "delivery_days", "promised_days", "freight_ratio", "log_gmv", "n_items", "n_sellers",
           "installments", "same_state", "perfume_share"]
CATEGORICAL = ["region", "main_payment"]

# Themes in Portuguese review comments (keyword rules, accents removed). A comment can match several themes.
THEMES = {
    "Delivery late or missing": r"entreg|nao receb|ainda nao|cheg|prazo|atras|demor|correio|aguard",
    "Wrong or incomplete order": r"errad|diferent|falt|incomplet|apenas um|so veio|veio so|nao veio",
    "Product quality or authenticity": r"qualidade|defeit|quebr|vazad|falsif|original|danific|ruim",
    "Seller or support contact": r"contato|resposta|atendimento|loja|vendedor|reclam|devolu|reembols|estorno",
}


def beauty_orders(con) -> pd.DataFrame:
    df = con.execute("""
        SELECT * FROM fct_orders
        WHERE has_beauty AND status = 'delivered' AND delivered_at IS NOT NULL AND review_score IS NOT NULL
        ORDER BY order_id
    """).df()
    df["bad_review"] = (df["review_score"] <= 2).astype(int)
    df["freight_ratio"] = df["freight"] / df["gmv"].clip(lower=1)
    df["log_gmv"] = np.log1p(df["gmv"])
    df["same_state"] = df["same_state"].fillna(False).astype(int)
    df["installments"] = df["installments"].fillna(1)
    df["perfume_share"] = df["perfume_gmv"] / df["gmv"].clip(lower=1)
    df["region"] = df["customer_state"].map(REGION).fillna("Other")
    df["main_payment"] = df["main_payment"].fillna("unknown")
    df["month"] = pd.to_datetime(df["purchased_at"]).dt.to_period("M").astype(str)
    return df


def delay_effect(df: pd.DataFrame) -> pd.DataFrame:
    """Bad-review rate by delivery punctuality."""
    bins = [-np.inf, -7, 0, 3, 10, np.inf]
    labels = ["7+ days early", "0 to 7 days early", "1 to 3 days late", "4 to 10 days late", "over 10 days late"]
    g = df.assign(band=pd.cut(df["delay_days"], bins=bins, labels=labels)).groupby("band", observed=True)
    return pd.DataFrame({"orders": g.size(), "bad_review_pct": (100 * g["bad_review"].mean()).round(1),
                         "avg_score": g["review_score"].mean().round(2)}).reset_index()


def _matrix(df: pd.DataFrame, columns=None):
    X = pd.concat([df[NUMERIC].astype(float), pd.get_dummies(df[CATEGORICAL], dtype=float)], axis=1)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


def train_and_test(df: pd.DataFrame, split_month: str = "2018-04") -> dict:
    """Train on orders bought before split_month, test on the following months."""
    import lightgbm as lgb
    train, test = df[df["month"] < split_month], df[df["month"] >= split_month]
    Xtr = _matrix(train)
    Xte = _matrix(test, Xtr.columns)
    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    logit.fit(Xtr, train["bad_review"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
        gbm.fit(Xtr, train["bad_review"])
    out = {"train_orders": len(train), "test_orders": len(test), "test_bad_rate_pct": round(100 * test["bad_review"].mean(), 1)}
    scores = {}
    for name, m in (("logistic", logit), ("gbm", gbm)):
        p = m.predict_proba(Xte)[:, 1]
        scores[name] = p
        top = np.argsort(-p)[: int(0.1 * len(p))]
        out[name] = {"auc": round(roc_auc_score(test["bad_review"], p), 3),
                     "avg_precision": round(average_precision_score(test["bad_review"], p), 3),
                     "bad_rate_in_top10_pct": round(100 * test["bad_review"].values[top].mean(), 1),
                     "share_of_bad_reviews_caught_top10_pct": round(
                         100 * test["bad_review"].values[top].sum() / max(test["bad_review"].sum(), 1), 1)}
    imp = pd.Series(gbm.booster_.feature_importance("gain"), index=Xtr.columns)
    imp = (100 * imp / imp.sum()).sort_values(ascending=False).round(1)
    best = max(("logistic", "gbm"), key=lambda k: out[k]["auc"])
    out["best_model"] = best
    scored = test.assign(risk=scores[best])
    return {"metrics": out, "importance": imp.rename("gain_pct").reset_index().rename(columns={"index": "feature"}),
            "scored_test": scored, "models": {"logistic": logit, "gbm": gbm}, "columns": list(Xtr.columns)}


def _strip(text: str) -> str:
    import unicodedata
    t = unicodedata.normalize("NFD", str(text).lower())
    return "".join(ch for ch in t if unicodedata.category(ch) != "Mn")


def complaint_themes(df: pd.DataFrame) -> pd.DataFrame:
    """Share of bad reviews with a written comment that mention each theme."""
    bad = df[(df["bad_review"] == 1) & df["comment_message"].notna()]
    text = bad["comment_message"].map(_strip)
    rows = [{"theme": k, "comments": int(text.str.contains(rx).sum()),
             "share_pct": round(100 * text.str.contains(rx).mean(), 1)} for k, rx in THEMES.items()]
    return pd.DataFrame(rows).sort_values("share_pct", ascending=False).reset_index(drop=True).assign(
        base_comments=len(bad))


def comment_examples(df: pd.DataFrame, theme: str, n: int = 3) -> list[str]:
    bad = df[(df["bad_review"] == 1) & df["comment_message"].notna()]
    hits = bad[bad["comment_message"].map(_strip).str.contains(THEMES[theme])]
    return [re.sub(r"\s+", " ", t)[:140] for t in hits["comment_message"].head(n)]
