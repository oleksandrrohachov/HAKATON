"""Long-term (seasonal) AQI forecast on weekly means, checked with an 80/20 train/test split.

The biggest pattern in the data is the yearly heating-season cycle, which a 7-day model
cannot see. Here days are averaged into weeks and Holt-Winters learns a 52-week season.
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from forecast import Z_95, daily_series

WEEKS_PER_YEAR = 52
TRAIN_SHARE = 0.8                       # first 80% of weeks = training, last 20% = test
MIN_TRAIN_WEEKS = 2 * WEEKS_PER_YEAR    # the model needs two full years to learn the season
DAYS_PER_WEEK = 7
SEASONAL_FORECAST_WEEKS = 26            # half a year ahead
SEASONAL_MODELS = ["train_mean", "last_year", "seasonal_hw"]


def weekly_series(df: pd.DataFrame) -> pd.Series:
    """Weekly mean AQI (weeks start on Monday) of the latest continuous stretch of data.

    Only complete weeks are kept, so a partial first/last week does not distort the mean.
    """
    days = daily_series(df, max_days=None)
    if days.empty:
        return days
    weeks = days.resample("W-MON", label="left", closed="left").agg(["mean", "count"])
    return weeks.loc[weeks["count"] == DAYS_PER_WEEK, "mean"].asfreq("W-MON").rename("aqi")


def _fit(train: pd.Series):
    """Holt-Winters with a 52-week multiplicative season (level x yearly cycle, no trend).

    Multiplicative: summer = level x 0.6 rather than level - 25, so forecasts never drop to ~0.
    """
    if len(train) < MIN_TRAIN_WEEKS:
        raise ValueError(
            f"Для сезонної моделі потрібно щонайменше {MIN_TRAIN_WEEKS} тижнів безперервних "
            f"даних для навчання, а є {len(train)}."
        )
    model = ExponentialSmoothing(train, trend=None, seasonal="mul", seasonal_periods=WEEKS_PER_YEAR,
                                 initialization_method="heuristic")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # statsmodels convergence chatter
        return model.fit()


def seasonal_split(df: pd.DataFrame, train_share: float = TRAIN_SHARE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on the first ``train_share`` of weeks, forecast the rest in one go.

    Returns ``(train, test)``: ``train`` has ``week, aqi``; ``test`` has ``week, actual`` and
    one column per model: ``train_mean`` (average of the training part), ``last_year``
    (repeat the last training year) and ``seasonal_hw`` (the seasonal model).
    """
    weeks = weekly_series(df)
    cut = int(len(weeks) * train_share)
    train, test = weeks.iloc[:cut], weeks.iloc[cut:]
    fit = _fit(train)
    horizon = len(test)
    predictions = {
        "train_mean": np.full(horizon, train.mean()),
        "last_year": np.resize(train.to_numpy()[-WEEKS_PER_YEAR:], horizon),
        "seasonal_hw": np.clip(np.asarray(fit.forecast(horizon)), 0, None),
    }
    return (
        pd.DataFrame({"week": train.index, "aqi": train.to_numpy()}),
        pd.DataFrame({"week": test.index, "actual": test.to_numpy(), **predictions}),
    )


def seasonal_forecast(df: pd.DataFrame, weeks: int = SEASONAL_FORECAST_WEEKS) -> pd.DataFrame:
    """Forecast weekly AQI ``weeks`` ahead using all data: ``date, yhat, lower, upper``."""
    history = weekly_series(df)
    fit = _fit(history)
    yhat = np.clip(np.asarray(fit.forecast(weeks)), 0, None)
    margin = Z_95 * float(np.std(fit.resid, ddof=1))
    dates = pd.date_range(history.index[-1] + pd.Timedelta(weeks=1), periods=weeks, freq="W-MON")
    return pd.DataFrame({"date": dates, "yhat": yhat, "lower": np.clip(yhat - margin, 0, None),
                         "upper": yhat + margin})
