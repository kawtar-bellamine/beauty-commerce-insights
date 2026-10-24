"""Beauty Commerce Insights: an interactive decision cockpit on real beauty e-commerce data (Olist).

Run: streamlit run app/dashboard.py   (uses the full data if downloaded, otherwise the committed sample)
"""
from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bci import insights as I  # noqa: E402
from bci.data import RAW, TABLES  # noqa: E402
from bci.experience import comment_examples  # noqa: E402
from bci.pipeline import run  # noqa: E402

st.set_page_config(page_title="Beauty Commerce Insights", page_icon="✦", layout="wide")

INK, MUTED, PLUM = "#1d1d1b", "#6b6a64", "#6b2f4f"
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SEG_COLORS = [S1, S5, S3, S4, S2]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

st.markdown(f"""
<style>
  .block-container {{ padding-top: 1.4rem; max-width: 1320px; }}
  [data-testid="stHeader"] {{ background: transparent; }}
  .hero {{ position: relative; overflow: hidden; background: linear-gradient(120deg, #151413 0%, #33202c 50%, #6b2f4f 100%);
          color: #fff; border-radius: 20px; padding: 26px 30px; margin-bottom: 16px; }}
  .hero:after {{ content: ""; position: absolute; inset: 0; background-image: radial-gradient(rgba(255,255,255,.08) 1px,
          transparent 1px); background-size: 22px 22px; pointer-events: none; }}
  .hero .tag {{ font-size: 12px; letter-spacing: .1em; text-transform: uppercase; color: #f1c9db; font-weight: 700; }}
  .hero .t {{ font-size: 30px; font-weight: 800; letter-spacing: -0.02em; margin: 4px 0 2px; }}
  .hero .s {{ color: #ead9e2; font-size: 14px; }}
  .brief {{ background: #fff; border-radius: 16px; padding: 16px 20px; margin-bottom: 16px;
           box-shadow: 0 1px 2px rgba(16,24,40,.06), inset 0 0 0 1px rgba(11,11,11,.08); }}
  .brief h4 {{ margin: 0 0 8px; font-size: 14px; color: {PLUM}; letter-spacing: .04em; text-transform: uppercase; }}
  .brief ol {{ margin: 0; padding-left: 20px; }} .brief li {{ margin: 4px 0; color: {INK}; font-size: 14.5px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; margin: 0 0 16px; }}
  .kpi {{ background: #fff; border-radius: 14px; padding: 14px 16px; box-shadow: 0 1px 2px rgba(16,24,40,.06),
         inset 0 0 0 1px rgba(11,11,11,.08); }}
  .kpi .l {{ font-size: 12px; color: {MUTED}; font-weight: 600; }}
  .kpi .v {{ font-size: 26px; font-weight: 800; color: {INK}; letter-spacing: -0.02em; }}
  .kpi .n {{ font-size: 12px; color: #898781; }}
  .note {{ color: {MUTED}; font-size: 13.5px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; margin: 6px 0 18px; }}
  .card {{ background: #fff; border-radius: 16px; padding: 16px; box-shadow: 0 1px 2px rgba(16,24,40,.06),
          inset 0 0 0 1px rgba(11,11,11,.08); border-top: 5px solid var(--c); }}
  .card .name {{ font-weight: 800; font-size: 16px; color: {INK}; }}
  .card .who {{ font-size: 13px; color: {MUTED}; margin: 4px 0 10px; min-height: 36px; }}
  .card .big {{ display: flex; gap: 16px; margin-bottom: 8px; }}
  .card .big b {{ display: block; font-size: 22px; color: {INK}; }}
  .card .big span {{ font-size: 11.5px; color: {MUTED}; }}
  .card .facts {{ font-size: 12.5px; color: {INK}; line-height: 1.6; }}
  .card .play {{ margin-top: 10px; font-size: 13px; background: #f6f1f4; border-radius: 10px; padding: 8px 10px; color: {INK}; }}
  .card .play b {{ color: {PLUM}; }}
  .sim {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 8px 0 14px; }}
  .sim div {{ background: #fff; border-radius: 14px; padding: 12px 14px; box-shadow: inset 0 0 0 1px rgba(11,11,11,.08); }}
  .sim b {{ display: block; font-size: 24px; color: {INK}; }} .sim span {{ font-size: 12px; color: {MUTED}; }}
  .good {{ color: #0a6b0a !important; }}
  @media (max-width: 1100px) {{ .kpis, .sim {{ grid-template-columns: repeat(2, 1fr); }} }}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Building the warehouse and running the models...")
def load():
    full = all((RAW / f).exists() for f in TABLES.values())
    return run(sample=not full) | {"full": full}


@st.cache_data
def geo():
    return json.loads((ROOT / "data" / "geo" / "brazil_states.geojson").read_text())


o = load()
k, seg, risk = o["kpis"], o["segments"], o["risk"]
rm = risk["metrics"]
best = o["best_forecast_model"]
sc = o["forecast_scores"].set_index("model")
baseline = sc.drop(index=best)["weekly_wape_pct"].min()
orders = o["orders"]

st.markdown(f"""<div class="hero"><div class="tag">Beauty e-commerce · decision cockpit</div>
<div class="t">✦ Beauty Commerce Insights</div>
<div class="s">Health &amp; beauty and perfumery on a Brazilian marketplace · real public data (Olist) ·
{k['beauty_orders']:,} beauty orders, 2016 to 2018{'' if o['full'] else ' · SAMPLE DATA'}</div></div>""",
            unsafe_allow_html=True)

st.markdown('<div class="brief"><h4>What the data says</h4><ol>' +
            "".join(f"<li>{escape(b)}</li>" for b in I.brief(o)) + "</ol></div>", unsafe_allow_html=True)


def kpi(label, value, note):
    return f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div><div class="n">{note}</div></div>'


st.markdown('<div class="kpis">' + "".join([
    kpi("Beauty sales", f"{k['beauty_gmv_brl'] / 1e6:.2f} M BRL", f"{k['beauty_share_of_marketplace_gmv_pct']}% of the marketplace"),
    kpi("Customers", f"{k['beauty_customers']:,}", f"only {seg['repeat_rate_pct']}% bought twice"),
    kpi("Average review", f"{k['beauty_avg_review']} / 5", f"{k['beauty_late_pct']}% of deliveries late"),
    kpi("Forecast error", f"{sc.loc[best, 'weekly_wape_pct']:.0f}%", f"weekly WAPE, best baseline {baseline:.0f}%"),
    kpi("Bad reviews caught", f"{rm[rm['best_model']]['share_of_bad_reviews_caught_top10_pct']:.0f}%",
        "by calling the riskiest 10%"),
    kpi("Top segment", f"{seg['profile'].iloc[0]['spend_share']:.0f}% of spend",
        f"{seg['profile'].iloc[0]['segment']}, {seg['profile'].iloc[0]['customer_share']:.0f}% of customers"),
]) + "</div>", unsafe_allow_html=True)

tabs = st.tabs(["Market map", "Sales forecast", "Delivery and reviews", "Shopper personas", "Data quality"])

# ------------------------------------------------------------------ market map
with tabs[0]:
    stt = o["states"].copy()
    metrics = {"Beauty sales (BRL)": "gmv", "Orders": "orders", "Late deliveries (%)": "late_pct",
               "Average delivery time (days)": "delivery_days", "Average review": "avg_review"}
    c1, c2 = st.columns([3, 2])
    pick = c2.radio("Colour the map by", list(metrics), index=0)
    field = metrics[pick]
    shapes = alt.Data(values=geo()["features"])
    reverse = field in ("avg_review",)
    chart = alt.Chart(shapes).mark_geoshape(stroke="white", strokeWidth=0.8).transform_lookup(
        lookup="properties.state", from_=alt.LookupData(stt, "state", list(metrics.values()) + ["state"])
    ).encode(
        color=alt.Color(f"{field}:Q", title=pick, scale=alt.Scale(range=SEQ[::-1] if reverse else SEQ),
                        legend=alt.Legend(orient="bottom", gradientLength=300)),
        tooltip=[alt.Tooltip("properties.name:N", title="state"),
                 alt.Tooltip("gmv:Q", title="sales (BRL)", format=",.0f"), alt.Tooltip("orders:Q", format=","),
                 alt.Tooltip("late_pct:Q", title="late %", format=".1f"),
                 alt.Tooltip("delivery_days:Q", title="delivery days", format=".1f"),
                 alt.Tooltip("avg_review:Q", title="avg review", format=".2f")]
    ).project("mercator").properties(height=470)
    c1.altair_chart(chart, use_container_width=True)
    top = stt.sort_values("gmv", ascending=False).head(10)
    c2.altair_chart(alt.Chart(top).mark_bar(color=S1, cornerRadiusEnd=3).encode(
        y=alt.Y("state:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False)),
        x=alt.X("gmv:Q", title="beauty sales (BRL)"),
        tooltip=["state", alt.Tooltip("gmv:Q", format=",.0f"), "orders"]).properties(height=250,
                                                                                 title="Top 10 states"),
        use_container_width=True)
    corr = stt[["delivery_days", "avg_review"]].corr().iloc[0, 1]
    c2.markdown(f'<p class="note">São Paulo alone brings {100 * top.iloc[0]["gmv"] / stt["gmv"].sum():.0f}% of beauty '
                f'sales. Across states, slower delivery goes with lower reviews (correlation {corr:.2f}).</p>',
                unsafe_allow_html=True)
    m = o["monthly"].melt(id_vars=["month"], value_vars=["health_beauty", "perfumery"], var_name="category",
                          value_name="sales")
    st.altair_chart(alt.Chart(m).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("yearmonth(month):T", title=None), y=alt.Y("sales:Q", title="sales (BRL)", stack=True),
        color=alt.Color("category:N", scale=alt.Scale(domain=["health_beauty", "perfumery"], range=[S1, S5]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("yearmonth(month):T", title="month"), "category:N",
                 alt.Tooltip("sales:Q", format=",.0f")]).properties(height=260, title="Monthly beauty sales"),
        use_container_width=True)

# ------------------------------------------------------------------ forecast
with tabs[1]:
    left, right = st.columns([3, 2])
    s = o["series"]
    hist = pd.DataFrame({"day": s.index, "sales": s.values})
    hist = hist[hist["day"] >= hist["day"].max() - pd.Timedelta(days=120)]
    fc = o["forecast"]
    base = alt.Chart(hist).mark_line(color="#9a9a95", strokeWidth=1.3).encode(
        x=alt.X("day:T", title=None), y=alt.Y("sales:Q", title="BRL per day"),
        tooltip=[alt.Tooltip("day:T"), alt.Tooltip("sales:Q", format=",.0f")])
    band = alt.Chart(fc).mark_area(color=S1, opacity=0.15).encode(x="day:T", y="low:Q", y2="high:Q")
    line = alt.Chart(fc).mark_line(color=S1, strokeWidth=2.2, strokeDash=[6, 3]).encode(
        x="day:T", y="forecast:Q", tooltip=[alt.Tooltip("day:T"), alt.Tooltip("forecast:Q", format=",.0f"),
                                            alt.Tooltip("low:Q", format=",.0f"), alt.Tooltip("high:Q", format=",.0f")])
    left.altair_chart((base + band + line).properties(height=330, title="Last 120 days and the next 28 (80% band)"),
                      use_container_width=True)
    right.markdown("**Backtest: 28 days ahead, 8 rolling origins**")
    right.dataframe(o["forecast_scores"].rename(columns={"daily_wape_pct": "daily WAPE %",
                                                         "weekly_wape_pct": "weekly WAPE %", "bias_pct": "bias %"}),
                    hide_index=True, use_container_width=True)
    right.markdown(f'<p class="note">The {best} model learns each day\'s ratio to the recent sales level by weekday, '
                   f'horizon and promotion calendar, and cuts the weekly error from {baseline:.0f}% to '
                   f'{sc.loc[best, "weekly_wape_pct"]:.0f}%. Every forecast only sees data available on its date.</p>',
                   unsafe_allow_html=True)
    nxt = fc.assign(week=((fc["day"] - fc["day"].min()).dt.days // 7) + 1).groupby("week")[["forecast", "low", "high"]].sum()
    right.markdown("**Next four weeks (BRL)**")
    right.dataframe(nxt.round(0).astype(int).rename(columns={"forecast": "forecast", "low": "low (P10)", "high": "high (P90)"}),
                    use_container_width=True)

# ------------------------------------------------------------------ delivery and reviews
with tabs[2]:
    d = o["delay_effect"].assign(band=lambda x: x["band"].astype(str))
    a, b = st.columns([3, 2])
    col = alt.Chart(d).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("band:N", sort=list(d["band"]), title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("bad_review_pct:Q", title="orders rated 1 or 2 stars (%)"),
        color=alt.Color("band:N", scale=alt.Scale(domain=list(d["band"]), range=[S3, S3, S4, S2, S2]), legend=None),
        tooltip=["band", "orders", "bad_review_pct", "avg_score"])
    a.altair_chart((col + col.mark_text(dy=-8, color=INK).encode(text=alt.Text("bad_review_pct:Q", format=".0f")))
                   .properties(height=290, title="Bad reviews by delivery punctuality"), use_container_width=True)
    b.markdown("**What if we delivered better?**")
    kept = b.slider("Late orders that arrive on the promised day (%)", 0, 100, 50, 10)
    faster = b.slider("All deliveries faster by (days)", 0, 5, 0)
    w = I.delivery_what_if(orders, faster, kept)
    b.markdown(f"""<div class="sim">
      <div><b class="good">-{w['bad_reviews_avoided']}</b><span>bad reviews</span></div>
      <div><b>{w['late_pct_new']}%</b><span>late, now {w['late_pct_now']}%</span></div>
      <div><b>{w['avg_score_new']:.2f}</b><span>avg score, now {w['avg_score_now']:.2f}</span></div>
      <div><b>{100 * w['bad_reviews_avoided'] / max(w['bad_reviews_now'], 1):.0f}%</b><span>of bad reviews avoided</span></div>
    </div><p class="note">Estimated with the bad-review rate observed in each punctuality band. It sizes the
    opportunity; it does not prove causality.</p>""", unsafe_allow_html=True)

    st.markdown("**Today's call list: the riskiest delivered orders and why**")
    best_m = rm["best_model"]
    cl = I.call_list(risk["scored_test"], risk["models"]["logistic"], risk["columns"], n=12)
    st.markdown(f'<p class="note">A {best_m} model scores each order on delivery day (AUC {rm[best_m]["auc"]} on '
                f'{rm["test_orders"]:,} later orders). In this list, {cl["actually_bad"].mean():.0%} of the orders '
                f'really ended with a bad review, against {rm["test_bad_rate_pct"]}% on average.</p>',
                unsafe_allow_html=True)
    st.dataframe(cl.rename(columns={"risk_pct": "risk %", "next_action": "next best action",
                                    "actually_bad": "got a bad review"}), hide_index=True, use_container_width=True)

    a, b = st.columns([2, 3])
    th = o["themes"]
    a.markdown("**What unhappy customers write**")
    a.altair_chart(alt.Chart(th).mark_bar(color=S5, cornerRadiusEnd=3).encode(
        y=alt.Y("theme:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
        x=alt.X("share_pct:Q", title="share of written bad reviews (%)"),
        tooltip=["theme", "comments", "share_pct"]).properties(height=190), use_container_width=True)
    theme = b.selectbox("Read real comments (Portuguese)", list(th["theme"]))
    for ex in comment_examples(orders, theme, 4):
        b.markdown(f"> {escape(ex)}")

# ------------------------------------------------------------------ personas
with tabs[3]:
    st.markdown(f'<p class="note">Only {seg["repeat_rate_pct"]}% of beauty customers ordered twice, so RFM would put '
                f'nearly everyone in one box. Personas are built on how people buy (k-means, k = {seg["k"]}, best '
                f'silhouette), then turned into a playbook.</p>', unsafe_allow_html=True)
    cards = ""
    for i, p in enumerate(I.personas(seg["profile"])):
        cards += f"""<div class="card" style="--c:{SEG_COLORS[i % 5]}"><div class="name">{escape(p['segment'])}</div>
        <div class="who">{escape(p['who'])}</div>
        <div class="big"><div><b>{p['customer_share']:.0f}%</b><span>of customers</span></div>
        <div><b>{p['spend_share']:.0f}%</b><span>of spend</span></div></div>
        <div class="facts">Average spend {p['avg_spend']:.0f} BRL · unit price {p['unit_price']:.0f} BRL<br>
        {p['items_per_order']:.1f} items per order · freight {100 * p['freight_ratio']:.0f}% of basket<br>
        {p['installments']:.1f} instalments · review {p['avg_score']:.2f} / 5</div>
        <div class="play"><b>Play:</b> {escape(p['play'])}</div></div>"""
    st.markdown(f'<div class="cards">{cards}</div>', unsafe_allow_html=True)

    st.markdown("**Free-shipping threshold simulator**")
    a, b = st.columns([2, 3])
    t = a.slider("Free shipping from a basket of (BRL)", 50, 300, 150, 25)
    fs = I.free_shipping(orders, t)
    a.markdown(f"""<div class="sim" style="grid-template-columns: repeat(2, 1fr)">
      <div><b>{fs['eligible_pct']}%</b><span>of orders qualify</span></div>
      <div><b>{fs['freight_offered_pct_of_sales']}%</b><span>of sales given back as freight</span></div>
      <div><b>{fs['close_orders']:,}</b><span>orders within 20% of the threshold</span></div>
      <div><b>{fs['avg_gap_brl']:.0f} BRL</b><span>average gap to reach it</span></div>
    </div><p class="note">The orders just below the threshold are the upsell target: one more small product would
    take them over.</p>""", unsafe_allow_html=True)
    curve = I.free_shipping_curve(orders)
    long = curve.melt(id_vars="threshold", value_vars=["eligible_pct", "freight_offered_pct_of_sales", "close_pct"],
                      var_name="measure", value_name="pct")
    names = {"eligible_pct": "orders qualifying", "freight_offered_pct_of_sales": "freight given (% of sales)",
             "close_pct": "orders within 20% below"}
    long["measure"] = long["measure"].map(names)
    lines = alt.Chart(long).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("threshold:Q", title="threshold (BRL)"), y=alt.Y("pct:Q", title="%"),
        color=alt.Color("measure:N", scale=alt.Scale(domain=list(names.values()), range=[S1, S2, S3]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["threshold", "measure", alt.Tooltip("pct:Q", format=".1f")])
    rule = alt.Chart(pd.DataFrame({"t": [t]})).mark_rule(color=PLUM, strokeDash=[4, 4]).encode(x="t:Q")
    b.altair_chart((lines + rule).properties(height=300), use_container_width=True)

# ------------------------------------------------------------------ data quality
with tabs[4]:
    st.markdown("**Pipeline:** 8 raw CSV files (SHA-256 checked) → DuckDB → typed staging → star schema "
                "(`fct_orders`, `fct_order_item`, `dim_product`) → daily sales mart → models → this cockpit.")
    q = o["quality"]
    st.dataframe(q, hide_index=True, use_container_width=True,
                 column_config={"failing_pct": st.column_config.NumberColumn("failing %", format="%.3f")})
    st.markdown('<p class="note">Blocking checks stop the pipeline; warnings are handled in staging (latest review '
                'kept, missing categories labelled "unknown", incomplete final days cut before forecasting).</p>',
                unsafe_allow_html=True)
