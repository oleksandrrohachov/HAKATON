"""Tab "Прогноз": what will happen and how much to trust it."""
import streamlit as st

import charts
import forecast
from glossary import FORECAST_INTRO
from ui import data
from ui.widgets import chart

PERIOD_FORMAT = "%m.%Y"


def _short_term(city: str, days: int) -> None:
    """Daily forecast for the next ``days`` days and its rolling check."""
    st.subheader(f"1. Короткостроковий прогноз на {days} днів")
    try:
        result, backtest = data.short_forecast(city, days), data.short_backtest(city, days)
    except ValueError as exc:
        st.warning(str(exc))
        return
    chart(charts.line_with_forecast(data.daily(city), result, title=f"Прогноз AQI на {days} днів"), "short_forecast")
    errors = forecast.model_errors(backtest)
    mae = errors.set_index("model").loc["holt_winters", "mae"]
    st.success(f"Перевірка на 8 останніх тижнях: у середньому прогноз помиляється на {mae:.0f} одиниць AQI.")
    left, right = st.columns(2)
    chart(charts.model_comparison(errors, title="Помилка 7-денних моделей (MAE)"), "short_models", left)
    chart(charts.predicted_vs_actual(backtest, "holt_winters"), "short_scatter", right)


def _seasonal_check(city: str) -> bool:
    """80/20 train/test check of the seasonal model; returns False if it is impossible."""
    st.subheader("2. Сезонна модель: перевірка 80% навчання / 20% тест")
    try:
        train, test = data.seasonal_check(city)
    except ValueError as exc:
        st.info(str(exc))
        return False
    errors = forecast.model_errors(test).set_index("model")["mae"]
    cols = st.columns(4)
    for col, label, part in ((cols[0], "Навчання 80%, тижнів", train), (cols[1], "Тест 20%, тижнів", test)):
        col.metric(label, len(part))
        col.caption(f"{part['week'].min():{PERIOD_FORMAT}} – {part['week'].max():{PERIOD_FORMAT}}")
    cols[2].metric("MAE моделі", f"{errors['seasonal_hw']:.1f}")
    cols[3].metric("MAE «як рік тому»", f"{errors['last_year']:.1f}")
    chart(charts.train_test_chart(train, test), "train_test")
    left, right = st.columns(2)
    chart(charts.model_comparison(errors.rename_axis("model").reset_index(), title="Помилка на тесті 20% (MAE)"),
          "seasonal_models", left)
    chart(charts.predicted_vs_actual(test, "seasonal_hw"), "short_scatter", right)
    return True


def _seasonal_outlook(city: str) -> None:
    """Half-year weekly forecast by the checked seasonal model."""
    st.subheader("3. Прогноз за сезоном на пів року")
    history, outlook = data.seasonal_outlook(city)
    chart(charts.line_with_forecast(history, outlook, title="Тижневий AQI: історія та прогноз на 26 тижнів"),
          "seasonal_forecast")


def render(city: str, days: int) -> None:
    """Whole forecast tab."""
    st.markdown(FORECAST_INTRO)
    _short_term(city, days)
    st.divider()
    if _seasonal_check(city):
        _seasonal_outlook(city)
