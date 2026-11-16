import pandas as pd

import analysis


def test_daily_shape(hourly_df):
    result = analysis.daily(hourly_df)
    assert list(result.columns) == ["date", "pm25", "aqi"]
    assert len(result) == 120


def test_profiles_have_expected_keys(hourly_df):
    assert analysis.hourly_profile(hourly_df)["hour"].tolist() == list(range(24))
    assert analysis.weekday_profile(hourly_df)["weekday"].tolist() == list(range(7))


def test_matrix_shape_and_weekend_is_worse(hourly_df):
    matrix = analysis.hour_weekday_matrix(hourly_df)
    assert matrix.shape == (7, 24)
    assert matrix.loc[5].mean() > matrix.loc[1].mean()


def test_monthly_and_summary(hourly_df):
    assert {"year", "month", "pm25", "aqi"} <= set(analysis.monthly(hourly_df).columns)
    result = analysis.summary(hourly_df)
    assert result["last_time"] == hourly_df["time"].iloc[-1]
    assert result["worst_day_aqi"] >= result["mean_aqi_30d"]


def test_summary_on_empty_data(hourly_df):
    assert analysis.summary(hourly_df.iloc[:0])["last_aqi"] is None


def test_last_days_window(hourly_df):
    window = analysis.last_days(hourly_df, 10)
    assert window["time"].max() == hourly_df["time"].max()
    assert window["time"].min() > hourly_df["time"].max() - pd.Timedelta(days=10)


def test_weekly_and_yearly(hourly_df):
    weekly = analysis.weekly(hourly_df)
    assert list(weekly.columns) == ["week", "pm25", "aqi"] and 17 <= len(weekly) <= 19
    yearly = analysis.yearly(hourly_df)
    assert yearly.loc[0, "year"] == 2025 and yearly.loc[0, "days"] == 120


def test_month_year_matrix_has_12_month_columns(hourly_df):
    matrix = analysis.month_year_matrix(hourly_df)
    assert list(matrix.columns) == list(range(1, 13))
    assert matrix.loc[2025, [1, 2, 3, 4]].notna().all() and matrix.loc[2025, 12] != matrix.loc[2025, 12]


def test_city_stats_and_matrices(hourly_df):
    dfs = {"a": hourly_df, "b": hourly_df}
    stats = analysis.city_stats(dfs)
    assert stats["city"].tolist() == ["a", "b"]
    assert stats.loc[0, "best_hour"] in (17, 18, 19)  # minimum of the sin pattern (noise may shift by 1)
    assert stats.loc[0, "worst_hour"] in (5, 6, 7)
    assert analysis.city_hour_matrix(dfs).shape == (2, 24)
    long = analysis.daily_by_city(dfs)
    assert list(long.columns) == ["city", "date", "aqi"] and len(long) == 240


def test_city_helpers_on_empty_input():
    assert analysis.city_stats({}).empty
    assert analysis.city_hour_matrix({}).empty
    assert analysis.daily_by_city({}).empty
