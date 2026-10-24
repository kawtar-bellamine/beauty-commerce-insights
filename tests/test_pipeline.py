"""Tests on the committed sample of the real data (runs in seconds, no download)."""
import numpy as np
import pandas as pd
import pytest

from bci import experience as E
from bci import forecast as F
from bci import segments as S
from bci.data import SAMPLE
from bci.quality import run_checks
from bci.warehouse import build


@pytest.fixture(scope="module")
def con():
    return build(folder=SAMPLE)


def test_star_schema_is_consistent(con):
    n_orders = con.execute("SELECT COUNT(*) FROM raw_orders").fetchone()[0]
    assert con.execute("SELECT COUNT(*) FROM fct_orders").fetchone()[0] == n_orders
    assert con.execute("SELECT COUNT(*) - COUNT(DISTINCT order_id) FROM fct_orders").fetchone()[0] == 0
    raw_gmv = con.execute("SELECT SUM(CAST(price AS DOUBLE)) FROM raw_items").fetchone()[0]
    assert abs(con.execute("SELECT SUM(gmv) FROM fct_orders").fetchone()[0] - raw_gmv) < 1e-6
    beauty = con.execute("SELECT COUNT(*) FROM fct_orders WHERE has_beauty").fetchone()[0]
    assert beauty == 3500


def test_one_review_per_order(con):
    assert con.execute("SELECT COUNT(*) - COUNT(DISTINCT order_id) FROM stg_reviews").fetchone()[0] == 0


def test_quality_checks_have_no_blocking_failure(con):
    q = run_checks(con)
    assert len(q) == 12
    assert not ((q["severity"] == "blocking") & (q["status"] == "FAIL")).any()
    assert set(q["status"]) <= {"PASS", "WARN", "FAIL"}


def test_incomplete_tail_is_cut():
    days = pd.date_range("2018-01-01", periods=120, freq="D")
    orders = np.full(120, 40)
    orders[-5:] = [10, 8, 5, 3, 1]
    daily = pd.DataFrame({"day": days, "gmv": orders * 100.0, "orders": orders})
    s = F.complete_series(daily, start="2018-01-01")
    assert s.index[-1] == days[-6]


def test_black_friday_flag():
    cal = F.calendar(pd.date_range("2017-11-20", "2017-11-30"))
    flagged = cal.index[cal["black_friday"] == 1]
    assert pd.Timestamp("2017-11-24") in flagged and pd.Timestamp("2017-11-30") not in flagged


def test_backtest_uses_only_the_past(con):
    s = F.complete_series(con.execute("SELECT * FROM daily_beauty_sales").df())
    bt = F.backtest(s, n_origins=2)
    assert set(bt["model"]) == set(F.MODELS)
    assert (bt["day"] >= bt["origin"]).all()
    # Changing the future must not change a forecast made before it.
    s2 = s.copy()
    s2.iloc[-10:] *= 50
    bt2 = F.backtest(s2, n_origins=2)
    first = bt["origin"].min()
    a = bt[bt["origin"] == first].sort_values(["model", "day"])["forecast"].values
    b = bt2[bt2["origin"] == first].sort_values(["model", "day"])["forecast"].values
    np.testing.assert_allclose(a, b)


def test_wape():
    assert F.wape([10, 10], [10, 10]) == 0
    assert F.wape([10, 10], [0, 20]) == pytest.approx(1.0)


def test_late_deliveries_get_worse_reviews(con):
    d = E.delay_effect(E.beauty_orders(con)).set_index("band")
    assert d.loc["over 10 days late", "bad_review_pct"] > 3 * d.loc["7+ days early", "bad_review_pct"]


def test_review_risk_model_beats_chance(con):
    r = E.train_and_test(E.beauty_orders(con))
    m = r["metrics"]
    assert m[m["best_model"]]["auc"] > 0.6
    assert abs(r["importance"]["gain_pct"].sum() - 100) < 0.5


def test_complaint_themes(con):
    t = E.complaint_themes(E.beauty_orders(con))
    assert t.iloc[0]["theme"] == "Delivery late or missing"
    assert (t["share_pct"].between(0, 100)).all()


def test_segments_are_reproducible_and_complete(con):
    a, b = S.segment(con), S.segment(con)
    pd.testing.assert_frame_equal(a["profile"], b["profile"])
    assert a["profile"]["customer_share"].sum() == pytest.approx(100, abs=0.01)
    assert a["profile"]["spend_share"].sum() == pytest.approx(100, abs=0.01)
    assert a["profile"]["segment"].is_unique
    assert a["k"] in (3, 4, 5, 6)


def test_what_if_is_monotonic_and_deterministic(con):
    from bci import insights as I
    o = E.beauty_orders(con)
    curve = I.what_if_curve(o, 5)
    assert curve["bad_reviews_avoided"].iloc[0] == 0
    assert curve["bad_reviews_avoided"].is_monotonic_increasing
    a, b = I.delivery_what_if(o, 0, 50), I.delivery_what_if(o, 0, 50)
    assert a == b
    full = I.delivery_what_if(o, 0, 100)
    assert full["late_pct_new"] == 0 and full["bad_reviews_avoided"] > a["bad_reviews_avoided"] > 0


def test_free_shipping_threshold_trade_off(con):
    from bci import insights as I
    c = I.free_shipping_curve(E.beauty_orders(con))
    assert c["eligible_pct"].is_monotonic_decreasing
    assert c["freight_offered_brl"].is_monotonic_decreasing


def test_call_list_explains_each_order(con):
    from bci import insights as I
    r = E.train_and_test(E.beauty_orders(con))
    cl = I.call_list(r["scored_test"], r["models"]["logistic"], r["columns"], n=10)
    assert len(cl) == 10 and cl["risk_pct"].is_monotonic_decreasing
    assert cl["why"].str.len().gt(0).all() and cl["next_action"].str.len().gt(0).all()
