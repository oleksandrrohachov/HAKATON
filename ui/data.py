"""Cached data access for the Streamlit app (every heavy call is computed once per input)."""
import pandas as pd
import streamlit as st

import analysis
import forecast
import loader
import seasonal


@st.cache_data(show_spinner="Завантажуємо дані…")
def city_data(city: str) -> pd.DataFrame:
    """Hourly data of one city."""
    return loader.load_city(city)


def all_cities(cities: list[str]) -> dict[str, pd.DataFrame]:
    """Hourly data of every city, keyed by the Ukrainian city name."""
    return {loader.city_label(city): city_data(city) for city in cities}


@st.cache_data(show_spinner="Рахуємо прогноз…")
def short_forecast(city: str, days: int) -> pd.DataFrame:
    """Daily forecast for ``days`` days."""
    return forecast.forecast_aqi(city_data(city), days)


@st.cache_data(show_spinner="Перевіряємо прогноз на минулому…")
def short_backtest(city: str, days: int) -> pd.DataFrame:
    """Rolling backtest of the daily models."""
    return forecast.rolling_backtest(city_data(city), days)


@st.cache_data(show_spinner="Розкладаємо ряд на тренд і сезонність…")
def decomposition(city: str) -> pd.DataFrame:
    """Yearly decomposition of daily AQI."""
    return forecast.decompose_daily(city_data(city))


@st.cache_data(show_spinner="Навчаємо сезонну модель на 80% даних…")
def seasonal_check(city: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 train/test split of the seasonal model."""
    return seasonal.seasonal_split(city_data(city))


@st.cache_data(show_spinner="Рахуємо сезонний прогноз…")
def seasonal_outlook(city: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weekly history (``date, aqi``) and the 26-week seasonal forecast."""
    df = city_data(city)
    history = seasonal.weekly_series(df).rename_axis("date").reset_index()
    return history, seasonal.seasonal_forecast(df)


def daily(city: str) -> pd.DataFrame:
    """Daily means of one city."""
    return analysis.daily(city_data(city))
