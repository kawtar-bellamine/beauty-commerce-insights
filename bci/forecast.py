"""Daily beauty sales forecasting, 28 days ahead, scored by rolling-origin backtesting.

Four models, from the simplest to the most flexible:
  naive_7d   same weekday last week
  weekday_4w mean of the same weekday over the last 4 weeks
  ridge      ridge regression on trend, weekday and promotion calendar (log scale)
  gbm        LightGBM learning the daily ratio to the recent level, by horizon, weekday and calendar
Every forecast only uses data available at its origin.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

HORIZON = 28


def complete_series(daily: pd.DataFrame, start: str = "2017-01-01") -> pd.Series:
    """Continuous daily series, cut where the extract becomes incomplete (orders fall below half the usual)."""
    s = daily.set_index(pd.to_datetime(daily["day"]))["gmv"].asfreq("D", fill_value=0.0)
    s = s[s.index >= start]
    orders = daily.set_index(pd.to_datetime(daily["day"]))["orders"].reindex(s.index, fill_value=0)
    typical = orders.rolling(28, min_periods=14).median().shift(1)
    tail = orders < 0.5 * typical
    last_ok = s.index[-1]
    for day in reversed(s.index):  # walk back over the incomplete tail
        if not tail.loc[day]:
            last_ok = day
            break
    return s[s.index <= last_ok]


def calendar(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Brazilian e-commerce calendar: Black Friday week (fourth Friday of November, plus around it)."""
    bf = []
    for d in idx:
        nov = pd.Timestamp(d.year, 11, 1)
        friday = nov + pd.Timedelta(days=(4 - nov.weekday()) % 7 + 21)
        bf.append(-1 <= (d - friday).days <= 3)
    return pd.DataFrame({"weekday": idx.day_of_week, "dom": idx.day, "month": idx.month,
                         "black_friday": np.array(bf, dtype=int)}, index=idx)


def naive_7d(hist: pd.Series, days: pd.DatetimeIndex) -> np.ndarray:
    last_week = hist.iloc[-7:]
    return np.array([last_week[last_week.index.day_of_week == d.weekday()].iloc[0] for d in days])


def weekday_4w(hist: pd.Series, days: pd.DatetimeIndex) -> np.ndarray:
    tail = hist.iloc[-28:]
    m = tail.groupby(tail.index.day_of_week).mean()
    return np.array([m.loc[d.weekday()] for d in days])


def ridge(hist: pd.Series, days: pd.DatetimeIndex) -> np.ndarray:
    def X(idx):
        c = calendar(idx)
        t = (idx - hist.index[0]).days.values / 365.0
        wd = pd.get_dummies(c["weekday"].astype(pd.CategoricalDtype(range(7))), prefix="wd").values
        return np.column_stack([t, wd, c["black_friday"].values])
    m = Ridge(alpha=1.0).fit(X(hist.index), np.log1p(hist.values))
    return np.expm1(m.predict(X(days))).clip(min=0)


def _gbm_rows(series: pd.Series, origins, with_target: bool):
    rows = []
    span = pd.date_range(series.index[0], series.index[-1] + pd.Timedelta(days=HORIZON), freq="D")
    cal_all = calendar(span)
    for o in origins:
        past = series[series.index < o]
        if len(past) < 56:
            continue
        level = past.iloc[-28:].mean()
        wd_mean = past.iloc[-28:].groupby(past.iloc[-28:].index.day_of_week).mean()
        for h in range(1, HORIZON + 1):
            d = o + pd.Timedelta(days=h - 1)
            if with_target and d not in series.index:
                break
            cal = cal_all.loc[d]
            row = {"origin": o, "day": d, "h": h, "weekday": cal["weekday"], "dom": cal["dom"],
                   "black_friday": cal["black_friday"], "wd_ratio": wd_mean.get(d.weekday(), level) / max(level, 1),
                   "level": level, "trend": (past.iloc[-28:].mean() / max(past.iloc[-56:-28].mean(), 1))}
            if with_target:
                row["y"] = np.log1p(series.loc[d]) - np.log1p(level)
            rows.append(row)
    return pd.DataFrame(rows)


FEATURES = ["h", "weekday", "dom", "black_friday", "wd_ratio", "trend"]


def gbm(hist: pd.Series, days: pd.DatetimeIndex) -> np.ndarray:
    import lightgbm as lgb
    origins = hist.index[56:len(hist) - HORIZON + 1:3]
    train = _gbm_rows(hist, origins, with_target=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1, random_state=0)
        model.fit(train[FEATURES], train["y"])
    test = _gbm_rows(hist.reindex(hist.index.union(days)).fillna(0), [days[0]], with_target=False)
    level = test["level"].values
    return np.expm1(model.predict(test[FEATURES]) + np.log1p(level)).clip(min=0)[:len(days)]


MODELS = {"naive_7d": naive_7d, "weekday_4w": weekday_4w, "ridge": ridge, "gbm": gbm}


def wape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.abs(y - p).sum() / max(np.abs(y).sum(), 1e-9))


def backtest(series: pd.Series, n_origins: int = 8, step: int = 14) -> pd.DataFrame:
    """Forecasts from n_origins dates, every `step` days, ending HORIZON days before the end of the series."""
    last_origin = series.index[-HORIZON]
    origins = [last_origin - pd.Timedelta(days=step * k) for k in range(n_origins)][::-1]
    rows = []
    for o in origins:
        hist = series[series.index < o]
        days = pd.date_range(o, periods=HORIZON, freq="D")
        actual = series.reindex(days).values
        for name, fn in MODELS.items():
            pred = fn(hist, days)
            for d, a, pr, h in zip(days, actual, pred, range(1, HORIZON + 1)):
                rows.append({"origin": o, "day": d, "h": h, "model": name, "actual": a, "forecast": pr})
    return pd.DataFrame(rows)


def score(bt: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, g in bt.groupby("model"):
        weekly = g.assign(week=(g["h"] - 1) // 7).groupby(["origin", "week"])[["actual", "forecast"]].sum()
        out.append({"model": name, "daily_wape_pct": round(100 * wape(g["actual"], g["forecast"]), 1),
                    "weekly_wape_pct": round(100 * wape(weekly["actual"], weekly["forecast"]), 1),
                    "bias_pct": round(100 * (g["forecast"].sum() / g["actual"].sum() - 1), 1)})
    return pd.DataFrame(out).sort_values("weekly_wape_pct").reset_index(drop=True)


def forecast_ahead(series: pd.Series, model: str, bt: pd.DataFrame) -> pd.DataFrame:
    """Next HORIZON days with an 80% band taken from the model's own backtest errors."""
    days = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    pred = MODELS[model](series, days)
    g = bt[bt["model"] == model]
    ratio = (g["actual"] + 1) / (g["forecast"] + 1)
    lo, hi = np.quantile(ratio, [0.1, 0.9])
    return pd.DataFrame({"day": days, "forecast": pred, "low": pred * lo, "high": pred * hi})
