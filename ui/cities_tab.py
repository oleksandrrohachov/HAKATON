"""Tab "Міста": comparison of all cities."""
import streamlit as st

import analysis
import charts
from ui import data
from ui.widgets import chart

STATS_COLUMNS = {
    "city": "Місто", "mean_aqi": "Середній AQI", "median_aqi": "Медіана", "p95_aqi": "95-й перцентиль",
    "unhealthy_share": "Шкідливих годин, %", "best_hour": "Найчистіша година", "worst_hour": "Найбрудніша година",
}
HOUR_COLUMNS = ("Найчистіша година", "Найбрудніша година")


def render(cities: list[str]) -> None:
    """Bars, statistics table, distribution and hourly heatmap of all cities."""
    frames = data.all_cities(cities)
    chart(charts.compare_cities(frames), "compare_cities")
    st.subheader("Порівняння за останній рік")
    table = analysis.city_stats(frames).rename(columns=STATS_COLUMNS).round(1)
    for column in HOUR_COLUMNS:
        table[column] = table[column].map(lambda hour: f"{int(hour):02d}:00")
    st.dataframe(table, hide_index=True, width="stretch")
    chart(charts.city_boxplot(analysis.daily_by_city(frames)), "city_boxplot")
    chart(charts.heatmap_city_hour(analysis.city_hour_matrix(frames)), "city_hour")
