"""Tab "Аналітика": what has happened (measured data only)."""
import pandas as pd
import streamlit as st

import analysis
import charts
import insights
from glossary import ANALYTICS_INTRO
from ui import data
from ui.widgets import chart


def render(city: str, df: pd.DataFrame) -> None:
    """Findings and pattern charts of one city."""
    st.markdown(ANALYTICS_INTRO)
    st.subheader("Головне про це місто")
    for finding in insights.key_findings(df):
        st.markdown(f"- {finding}")
    month = int(df["time"].iloc[-1].month)
    st.markdown(f"- У {insights.MONTH_NAMES[month - 1]}: {insights.compare_to_last_year(df, month)}.")

    st.subheader("Протягом доби і тижня")
    left, right = st.columns(2)
    chart(charts.hourly_profile_chart(analysis.hourly_profile(df)), "hourly_profile", left)
    chart(charts.weekday_profile_chart(analysis.weekday_profile(df)), "weekday_profile", right)
    chart(charts.heatmap_hour_weekday(analysis.hour_weekday_matrix(df)), "heatmap_hour_weekday")

    st.subheader("Протягом року і за роками")
    chart(charts.heatmap_month_year(analysis.month_year_matrix(df)), "heatmap_month_year")
    chart(charts.category_share_chart(insights.category_share_by_year(df)), "category_share")
    chart(charts.weekly_trend(analysis.weekly(df)), "weekly_trend")
    try:
        chart(charts.decomposition_chart(data.decomposition(city)), "decomposition")
    except ValueError as exc:
        st.info(str(exc))
