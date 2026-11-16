import pytest

import forecast


def test_forecast_shape_and_corridor(hourly_df):
    result = forecast.forecast_aqi(hourly_df, days=7)
    assert len(result) == 7
    assert list(result.columns) == ["date", "yhat", "lower", "upper"]
    assert ((result["lower"] <= result["yhat"]) & (result["yhat"] <= result["upper"])).all()
    assert (result["lower"] >= 0).all()


def test_forecast_without_seasonality_on_short_history(hourly_df):
    short = hourly_df[hourly_df["time"] < "2025-01-21"]  # 20 days < 28
    assert len(forecast.forecast_aqi(short, days=5)) == 5


def test_too_little_data_raises(hourly_df):
    with pytest.raises(ValueError):
        forecast.forecast_aqi(hourly_df.head(24 * 3))


def test_backtest_mae_is_small_on_regular_data(hourly_df):
    assert 0 <= forecast.backtest_mae(hourly_df, days=7) < 5


def test_rolling_backtest_and_model_errors(hourly_df):
    backtest = forecast.rolling_backtest(hourly_df, days=7, folds=4)
    assert list(backtest.columns) == ["date", "actual", *forecast.MODELS]
    assert len(backtest) == 28
    errors = forecast.model_errors(backtest)
    assert set(errors["model"]) == set(forecast.MODELS)
    assert errors["mae"].is_monotonic_increasing and (errors["mae"] >= 0).all()


def test_decompose_daily_weekly_period(hourly_df):
    parts = forecast.decompose_daily(hourly_df, period=7)
    assert list(parts.columns) == ["date", "observed", "trend", "seasonal", "resid"]
    assert parts["trend"].notna().all()


def test_decompose_needs_two_periods(hourly_df):
    with pytest.raises(ValueError):
        forecast.decompose_daily(hourly_df, period=365)


def test_model_errors_uses_every_model_column():
    import pandas as pd
    table = pd.DataFrame({"week": [1, 2], "actual": [10.0, 20.0], "good": [11.0, 19.0], "bad": [0.0, 0.0]})
    assert forecast.model_errors(table)["model"].tolist() == ["good", "bad"]
