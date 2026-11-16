import pandas as pd
import plotly.graph_objects as go

import analysis
import charts
import charts.base
import forecast
import insights


def test_charts_return_figures(hourly_df):
    daily = analysis.daily(hourly_df)
    figures = [
        charts.line_with_forecast(daily, forecast.forecast_aqi(hourly_df)),
        charts.heatmap_hour_weekday(analysis.hour_weekday_matrix(hourly_df)),
        charts.compare_cities({"a": hourly_df, "b": hourly_df}),
    ]
    assert all(isinstance(fig, go.Figure) and len(fig.data) > 0 for fig in figures)


def test_charts_survive_empty_data(hourly_df):
    empty = hourly_df.iloc[:0]
    figures = [
        charts.line_with_forecast(analysis.daily(empty), pd.DataFrame()),
        charts.heatmap_hour_weekday(pd.DataFrame()),
        charts.compare_cities({}),
        charts.compare_cities({"a": empty}),
    ]
    assert all(isinstance(fig, go.Figure) and len(fig.data) == 0 for fig in figures)


def test_color_for_aqi_is_consistent():
    assert charts.color_for_aqi(10) != charts.color_for_aqi(250)
    assert charts.color_for_aqi(float("nan")) == charts.NO_DATA_COLOR


def test_new_charts_return_figures(hourly_df):
    dfs = {"a": hourly_df, "b": hourly_df}
    backtest = forecast.rolling_backtest(hourly_df, folds=3)
    figures = [
        charts.weekly_trend(analysis.weekly(hourly_df)),
        charts.hourly_profile_chart(analysis.hourly_profile(hourly_df)),
        charts.weekday_profile_chart(analysis.weekday_profile(hourly_df)),
        charts.heatmap_month_year(analysis.month_year_matrix(hourly_df)),
        charts.heatmap_city_hour(analysis.city_hour_matrix(dfs)),
        charts.category_share_chart(insights.category_share_by_year(hourly_df)),
        charts.city_boxplot(analysis.daily_by_city(dfs)),
        charts.decomposition_chart(forecast.decompose_daily(hourly_df, period=7)),
        charts.predicted_vs_actual(backtest),
        charts.model_comparison(forecast.model_errors(backtest)),
    ]
    assert all(isinstance(fig, go.Figure) and len(fig.data) > 0 for fig in figures)


def test_new_charts_survive_empty_data():
    empty = pd.DataFrame()
    figures = [
        charts.weekly_trend(pd.DataFrame(columns=["week", "aqi"])),
        charts.hourly_profile_chart(pd.DataFrame(columns=["hour", "aqi"])),
        charts.weekday_profile_chart(pd.DataFrame(columns=["weekday", "aqi"])),
        charts.heatmap_month_year(empty), charts.heatmap_city_hour(empty), charts.category_share_chart(empty),
        charts.city_boxplot(empty), charts.decomposition_chart(empty), charts.predicted_vs_actual(empty),
        charts.model_comparison(empty),
    ]
    assert all(isinstance(fig, go.Figure) and len(fig.data) == 0 for fig in figures)


def test_train_test_chart():
    train = pd.DataFrame({"week": pd.date_range("2024-01-01", periods=8, freq="W-MON"), "aqi": range(8)})
    test = pd.DataFrame({"week": pd.date_range("2024-02-26", periods=2, freq="W-MON"), "actual": [5.0, 6.0],
                         "seasonal_hw": [5.5, 6.5]})
    assert len(charts.train_test_chart(train, test).data) == 3
    assert len(charts.train_test_chart(train.iloc[:0], test).data) == 0
