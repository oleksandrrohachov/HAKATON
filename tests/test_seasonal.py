import pytest

import forecast
import seasonal


def test_weekly_series_has_only_complete_weeks(yearly_df):
    weeks = seasonal.weekly_series(yearly_df)
    assert weeks.index.freqstr.startswith("W")
    assert (weeks.index.day_of_week == 0).all() and weeks.notna().all()


def test_split_is_80_20_and_in_time_order(yearly_df):
    train, test = seasonal.seasonal_split(yearly_df)
    total = len(train) + len(test)
    assert len(train) == int(total * 0.8)
    assert train["week"].max() < test["week"].min()
    assert list(test.columns) == ["week", "actual", *seasonal.SEASONAL_MODELS]


def test_seasonal_model_beats_plain_average(yearly_df):
    _, test = seasonal.seasonal_split(yearly_df)
    errors = forecast.model_errors(test).set_index("model")["mae"]
    assert errors["seasonal_hw"] < errors["train_mean"]


def test_seasonal_forecast_shape(yearly_df):
    result = seasonal.seasonal_forecast(yearly_df, weeks=10)
    assert len(result) == 10
    assert ((result["lower"] <= result["yhat"]) & (result["yhat"] <= result["upper"])).all()


def test_too_short_history_raises(hourly_df):
    with pytest.raises(ValueError):
        seasonal.seasonal_split(hourly_df)
