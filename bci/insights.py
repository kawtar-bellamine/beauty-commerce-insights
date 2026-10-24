"""Decision tools built on the analyses: what-if simulators, a call list and persona playbooks.

Everything here is computed from observed rates in the data. The simulators say what would happen if the observed
relationships held, they do not claim causality: they size an opportunity, they do not prove it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BANDS = [-np.inf, -7, 0, 3, 10, np.inf]
BAND_LABELS = ["7+ days early", "0 to 7 days early", "1 to 3 days late", "4 to 10 days late", "over 10 days late"]


# ------------------------------------------------------------------ delivery what-if

def band_rates(orders: pd.DataFrame) -> pd.Series:
    band = pd.cut(orders["delay_days"], bins=BANDS, labels=BAND_LABELS)
    return orders.groupby(band, observed=False)["bad_review"].mean()


def delivery_what_if(orders: pd.DataFrame, days_faster: float = 0, promise_kept_pct: float = 0) -> dict:
    """Deliveries arrive `days_faster` days earlier, and `promise_kept_pct` % of the remaining late orders arrive on
    the promised day. Bad reviews are re-estimated with the observed bad-review rate of each punctuality band."""
    rates = band_rates(orders)
    delay = orders["delay_days"] - days_faster
    late = delay > 0
    # Deterministic pick of which late orders are fixed (order ids are hexadecimal).
    fixed = late & (orders["order_id"].str[:8].map(lambda h: int(h, 16) % 100) < promise_kept_pct)
    delay = delay.where(~fixed, 0.0)
    shifted = pd.cut(delay, bins=BANDS, labels=BAND_LABELS)
    expected = shifted.map(rates).astype(float)
    base = float(orders["bad_review"].sum())
    new = float(expected.sum())
    late_now = float((orders["delay_days"] > 0).mean())
    late_new = float((delay > 0).mean())
    # Average score moves with the share of bad reviews, using the observed mean score of good and bad reviews.
    good_mean = orders.loc[orders["bad_review"] == 0, "review_score"].mean()
    bad_mean = orders.loc[orders["bad_review"] == 1, "review_score"].mean()
    n = len(orders)
    score_new = (new * bad_mean + (n - new) * good_mean) / n
    return {"days_faster": days_faster, "promise_kept_pct": promise_kept_pct, "orders": n, "bad_reviews_now": int(base), "bad_reviews_new": round(new),
            "bad_reviews_avoided": round(base - new), "late_pct_now": round(100 * late_now, 1),
            "late_pct_new": round(100 * late_new, 1), "avg_score_now": round(float(orders["review_score"].mean()), 2),
            "avg_score_new": round(float(score_new), 2)}


def what_if_curve(orders: pd.DataFrame, max_days: int = 7) -> pd.DataFrame:
    return pd.DataFrame([delivery_what_if(orders, d) for d in range(0, max_days + 1)])


# ------------------------------------------------------------------ free-shipping threshold

def free_shipping(orders: pd.DataFrame, threshold: float, near: float = 0.2) -> dict:
    """Free shipping above a basket threshold: who qualifies, what it costs, and who is close enough to upsell."""
    gmv, freight = orders["gmv"], orders["freight"]
    eligible = gmv >= threshold
    close = (gmv < threshold) & (gmv >= (1 - near) * threshold)
    return {"threshold": threshold, "eligible_pct": round(100 * eligible.mean(), 1),
            "freight_offered_brl": round(float(freight[eligible].sum())),
            "freight_offered_pct_of_sales": round(100 * float(freight[eligible].sum()) / float(gmv.sum()), 1),
            "close_orders": int(close.sum()), "close_pct": round(100 * close.mean(), 1),
            "avg_gap_brl": round(float((threshold - gmv[close]).mean()), 1) if close.any() else 0.0}


def free_shipping_curve(orders: pd.DataFrame, thresholds=range(50, 301, 25)) -> pd.DataFrame:
    return pd.DataFrame([free_shipping(orders, t) for t in thresholds])


# ------------------------------------------------------------------ call list

REASONS = {
    "delay_days": "delivered {delay_days:.0f} days after the promised date",
    "delivery_days": "took {delivery_days:.0f} days to arrive",
    "promised_days": "promised in {promised_days:.0f} days",
    "freight_ratio": "freight was {freight_pct:.0f}% of the basket",
    "n_sellers": "items from {n_sellers:.0f} different sellers",
    "n_items": "{n_items:.0f} items in the order",
    "log_gmv": "basket of {gmv:.0f} BRL",
}


def call_list(scored: pd.DataFrame, model, columns, n: int = 15) -> pd.DataFrame:
    """Riskiest orders with the two strongest reasons, from the logistic model's contributions."""
    from .experience import _matrix
    top = scored.sort_values("risk", ascending=False).head(n).copy()
    X = _matrix(top, columns)
    scaler, lr = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    contrib = pd.DataFrame(scaler.transform(X) * lr.coef_[0], columns=columns, index=top.index)
    rows = []
    for idx, r in top.iterrows():
        c = contrib.loc[idx, [k for k in REASONS if k in contrib.columns]].sort_values(ascending=False)
        ctx = r.to_dict() | {"freight_pct": 100 * r["freight_ratio"]}
        reasons = [REASONS[k].format(**ctx) for k, v in c.items() if v > 0][:2]
        late = r["delay_days"] > 0
        action = ("Call, apologise for the delay and offer a voucher" if late and r["delay_days"] > 3 else
                  "Check the parcel arrived complete" if r["n_sellers"] > 1 or r["n_items"] > 2 else
                  "Send a care message and a usage tip" if not late else "Send an apology message")
        rows.append({"order": r["order_id"][:8], "state": r["customer_state"], "risk_pct": round(100 * r["risk"]),
                     "why": "; ".join(reasons) or "unusual order profile", "next_action": action,
                     "actually_bad": bool(r["bad_review"])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ personas

PLAYBOOK = {
    "Premium seekers": ("Highest baskets, paid in many instalments.",
                        "Protect them: priority delivery and proactive tracking; promote prestige sets with "
                        "interest-free instalments."),
    "Everyday essentials": ("The largest group, one mid-price product at a time.",
                            "Grow the basket: bundles and 'complete your routine' suggestions at checkout."),
    "Fragrance buyers": ("Almost only perfumery, often a gift.",
                         "Sampling and discovery sets; gift wrapping and reminders before key dates."),
    "Freight-sensitive small baskets": ("Cheap items where shipping weighs most of the bill.",
                                        "Free-shipping threshold to lift them into a bigger basket."),
    "Routine stockers": ("Several items per order, the least satisfied segment.",
                         "Fix fulfilment first: single-parcel delivery and completeness check on multi-item orders."),
}


def personas(profile: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in profile.iterrows():
        base = r["segment"].replace(" (2)", "")
        who, play = PLAYBOOK.get(base, ("", ""))
        out.append({"segment": r["segment"], "who": who, "play": play,
                    "customer_share": r["customer_share"], "spend_share": r["spend_share"],
                    "avg_spend": r["avg_spend"], "unit_price": r["unit_price"], "items_per_order": r["items_per_order"],
                    "freight_ratio": r["freight_ratio"], "installments": r["installments"], "avg_score": r["avg_score"]})
    return out


# ------------------------------------------------------------------ executive brief

def brief(o: dict) -> list[str]:
    k, seg, risk = o["kpis"], o["segments"], o["risk"]["metrics"]
    sc = o["forecast_scores"].set_index("model")
    best = o["best_forecast_model"]
    top = seg["profile"].iloc[0]
    wi = delivery_what_if(o["orders"], promise_kept_pct=100)
    m = risk[risk["best_model"]]
    m_ = o["monthly"].assign(q=pd.to_datetime(o["monthly"]["month"]).dt.to_period("Q"))
    q = m_.groupby("q").agg(sales=("health_beauty", "sum"), p=("perfumery", "sum"), n=("month", "size"))
    q = (q["sales"] + q["p"])[q["n"] == 3]
    growth = (f"grew every quarter from {q.index[0]} to {q.index[-1]} ({q.iloc[-1] / q.iloc[0]:.1f}x)"
              if q.is_monotonic_increasing else f"went from {q.iloc[0] / 1e3:.0f}k to {q.iloc[-1] / 1e3:.0f}k BRL a quarter")
    return [
        f"Beauty is {k['beauty_share_of_marketplace_gmv_pct']}% of marketplace sales and {growth}; the "
        f"next 28 days are forecast with a weekly error of {sc.loc[best, 'weekly_wape_pct']:.0f}%.",
        f"Delivery decides satisfaction: if every late order had arrived on the promised day, about "
        f"{wi['bad_reviews_avoided']} of {wi['bad_reviews_now']} bad reviews "
        f"({100 * wi['bad_reviews_avoided'] / wi['bad_reviews_now']:.0f}%) would not have been written, and the average "
        f"score would rise from {wi['avg_score_now']} to {wi['avg_score_new']}.",
        f"{top['segment']} are {top['customer_share']:.0f}% of customers but {top['spend_share']:.0f}% of spend: "
        f"they deserve the best delivery promise.",
        f"Calling the riskiest 10% of delivered orders reaches {m['share_of_bad_reviews_caught_top10_pct']:.0f}% of "
        f"future bad reviews before they are written.",
    ]
