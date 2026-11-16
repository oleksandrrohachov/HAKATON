"""Top card "Зараз": the latest measurement and what to do about it."""
import pandas as pd
import streamlit as st

import analysis
import charts
import insights


def render(df: pd.DataFrame) -> None:
    """Latest AQI, its category, advice and the best hour to go outside."""
    info = analysis.summary(df)
    aqi = float("nan") if info["last_aqi"] is None else info["last_aqi"]
    st.subheader("Зараз")
    col_aqi, col_category, col_hour = st.columns(3)
    col_aqi.metric("AQI", "—" if pd.isna(aqi) else f"{aqi:.0f}")
    col_category.markdown(
        f"**Категорія**<br><span style='color:{charts.color_for_aqi(aqi)};"
        f"font-size:1.6rem;font-weight:700'>{insights.aqi_category(aqi)}</span>",
        unsafe_allow_html=True,
    )
    col_hour.metric("Найкраща година для вулиці", insights.best_hour(analysis.hourly_profile(df)))
    st.info(insights.advice(aqi))
    if info["last_time"] is not None:
        st.caption(f"Останнє вимірювання: {info['last_time']:%d.%m.%Y %H:%M}. "
                   f"Середній AQI за 30 днів: {info['mean_aqi_30d']:.0f}.")
