"""Full study on the real Olist data: writes results/ (summary.json, CSV tables, figures). About 20 seconds."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bci.data import download  # noqa: E402
from bci import insights as I  # noqa: E402
from bci.pipeline import run  # noqa: E402

OUT = ROOT / "results"
FIG = OUT / "figures"
INK, MUTED, GRID = "#1d1d1b", "#6b6a64", "#e4e3de"
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def title(fig, t):
    fig.suptitle(t, x=0.01, ha="left", fontsize=11, color=INK)


def fig_monthly(m: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    x = pd.to_datetime(m["month"])
    ax.bar(x, m["health_beauty"] / 1000, width=24, color=S1, label="health & beauty")
    ax.bar(x, m["perfumery"] / 1000, width=24, bottom=m["health_beauty"] / 1000, color=S5, label="perfumery",
           edgecolor="white", linewidth=1)
    ax.set_ylabel("sales, thousand BRL", color=MUTED, fontsize=9)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    nov = m[pd.to_datetime(m["month"]).dt.month.eq(11) & pd.to_datetime(m["month"]).dt.year.eq(2017)]
    if len(nov):
        ax.annotate("Black Friday", (pd.to_datetime(nov["month"].iloc[0]), (nov["health_beauty"] + nov["perfumery"]).iloc[0] / 1000),
                    xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8, color=MUTED)
    title(fig, "Beauty sales per month on the marketplace, January 2017 to August 2018")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_monthly_sales.png", dpi=150)
    plt.close(fig)


def fig_forecast(series: pd.Series, bt: pd.DataFrame, best: str, fc: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    recent = series[series.index >= series.index[-1] - pd.Timedelta(days=150)]
    ax.plot(recent.index, recent.values / 1000, color="#9a9a95", linewidth=1, label="actual daily sales")
    g = bt[bt["model"] == best]
    last = g[g["origin"] == g["origin"].max()]
    ax.plot(last["day"], last["forecast"] / 1000, color=S1, linewidth=2, label=f"backtest forecast ({best})")
    ax.fill_between(fc["day"], fc["low"] / 1000, fc["high"] / 1000, color=S1, alpha=0.15, linewidth=0)
    ax.plot(fc["day"], fc["forecast"] / 1000, color=S1, linewidth=2, linestyle="--", label="next 28 days, 80% band")
    ax.set_ylabel("thousand BRL per day", color=MUTED, fontsize=9)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
    title(fig, "Daily beauty sales: last backtest window and 28-day forecast")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_forecast.png", dpi=150)
    plt.close(fig)


def fig_delay(d: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    colors = [S3, S3, S4, S2, S2]
    bars = ax.bar(d["band"].astype(str), d["bad_review_pct"], color=colors, width=0.6)
    for b, v, n in zip(bars, d["bad_review_pct"], d["orders"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9, color=INK)
        ax.text(b.get_x() + b.get_width() / 2, 2, f"{n:,} orders", ha="center", fontsize=7.5, color="white")
    ax.set_ylabel("orders rated 1 or 2 stars (%)", color=MUTED, fontsize=9)
    ax.set_ylim(0, 90)
    style(ax)
    ax.tick_params(axis="x", labelsize=8)
    title(fig, "Real data: a late delivery multiplies bad reviews on beauty orders")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_delivery_vs_reviews.png", dpi=150)
    plt.close(fig)


def fig_segments(p: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    y = np.arange(len(p))[::-1]
    ax.barh(y + 0.2, p["customer_share"], height=0.38, color="#b9c3d3", label="share of customers")
    ax.barh(y - 0.2, p["spend_share"], height=0.38, color=S1, label="share of spend")
    ax.set_yticks(y, p["segment"])
    for yy, v in zip(y, p["spend_share"]):
        ax.text(v + 0.8, yy - 0.2, f"{v:.0f}%", va="center", fontsize=8, color=INK)
    ax.set_xlabel("%", color=MUTED, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    title(fig, "Shopper segments of beauty customers: who brings the revenue")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_segments.png", dpi=150)
    plt.close(fig)


def to_py(o):
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if hasattr(o, "item"):
        return o.item()
    return o


def main():
    print(download())
    FIG.mkdir(parents=True, exist_ok=True)
    o = run(OUT / "warehouse.duckdb")
    o["quality"].to_csv(OUT / "data_quality.csv", index=False)
    o["forecast_scores"].to_csv(OUT / "forecast_backtest_scores.csv", index=False)
    o["forecast"].to_csv(OUT / "forecast_next_28_days.csv", index=False)
    o["delay_effect"].to_csv(OUT / "delivery_vs_reviews.csv", index=False)
    o["themes"].to_csv(OUT / "complaint_themes.csv", index=False)
    o["risk"]["importance"].to_csv(OUT / "review_risk_drivers.csv", index=False)
    o["segments"]["profile"].to_csv(OUT / "segments.csv", index=False)
    o["states"].to_csv(OUT / "beauty_by_state.csv", index=False)
    seg = o["segments"]
    wi = I.what_if_curve(o["orders"])
    wi.to_csv(OUT / "what_if_faster_delivery.csv", index=False)
    kept = I.delivery_what_if(o["orders"], promise_kept_pct=100)
    fs = I.free_shipping_curve(o["orders"])
    fs.to_csv(OUT / "free_shipping_thresholds.csv", index=False)
    summary = {
        "data": {"source": "Olist public e-commerce data (Brazil), MIT licence", "orders": 99441,
                 "beauty_categories": ["health_beauty", "perfumery"]},
        "kpis": o["kpis"],
        "data_quality": o["quality"][["check", "status", "rows_failing", "description"]].to_dict(orient="records"),
        "forecast": {"series_end": str(o["series"].index[-1].date()), "horizon_days": 28,
                     "best_model": o["best_forecast_model"], "scores": o["forecast_scores"].to_dict(orient="records")},
        "review_risk": o["risk"]["metrics"],
        "delivery_vs_reviews": o["delay_effect"].assign(band=o["delay_effect"]["band"].astype(str)).to_dict(orient="records"),
        "complaint_themes": o["themes"].to_dict(orient="records"),
        "brief": I.brief(o),
        "what_if_promise_kept": kept,
        "segments": {"k": seg["k"], "silhouette": seg["silhouette"], "repeat_rate_pct": seg["repeat_rate_pct"],
                     "profile": seg["profile"].to_dict(orient="records")},
    }
    (OUT / "summary.json").write_text(json.dumps(to_py(summary), indent=1, default=str))
    fig_monthly(o["monthly"])
    fig_forecast(o["series"], o["backtest"], o["best_forecast_model"], o["forecast"])
    fig_delay(o["delay_effect"])
    fig_segments(seg["profile"])
    print(json.dumps(to_py({k: summary[k] for k in ("kpis", "forecast", "review_risk")}), indent=1, default=str))


if __name__ == "__main__":
    main()
