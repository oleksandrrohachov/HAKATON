"""Shared synthetic data: short hourly history (120 days) and a long one (4 years)."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def hourly_df() -> pd.DataFrame:
    times = pd.date_range("2025-01-01", periods=120 * 24, freq="h")
    rng = np.random.default_rng(0)
    aqi = 50 + 10 * np.sin(2 * np.pi * times.hour / 24) + 5 * (times.day_of_week >= 5) + rng.normal(0, 2, len(times))
    return pd.DataFrame({"time": times, "pm25": aqi / 3, "aqi": aqi})


@pytest.fixture
def yearly_df() -> pd.DataFrame:
    """Four years of hourly data with a strong winter peak (like the heating season)."""
    times = pd.date_range("2021-01-04", periods=4 * 365 * 24, freq="h")
    rng = np.random.default_rng(1)
    aqi = 45 + 20 * np.cos(2 * np.pi * times.day_of_year / 365.25) + rng.normal(0, 3, len(times))
    return pd.DataFrame({"time": times, "pm25": aqi / 3, "aqi": aqi})
