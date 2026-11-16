"""City comparison charts."""
import pandas as pd
import plotly.graph_objects as go

from analysis import last_days
from charts.base import LAYOUT, aqi_heatmap, color_for_aqi, empty_figure

COMPARE_DAYS = 30


def compare_cities(dfs: dict[str, pd.DataFrame]) -> go.Figure:
    """Bars of mean AQI over the last 30 days of each city's data."""
    means = {city: last_days(df, COMPARE_DAYS)["aqi"].mean() for city, df in dfs.items() if not df.empty}
    means = {city: value for city, value in means.items() if pd.notna(value)}
    if not means:
        return empty_figure()
    cities, values = zip(*sorted(means.items(), key=lambda item: item[1], reverse=True))
    fig = go.Figure(go.Bar(x=list(cities), y=list(values), marker_color=[color_for_aqi(v) for v in values],
                           text=[f"{v:.0f}" for v in values], textposition="outside"))
    return fig.update_layout(title=f"Середній AQI за останні {COMPARE_DAYS} днів", yaxis_title="AQI",
                             xaxis_title="Місто", **LAYOUT)


def city_boxplot(daily_by_city: pd.DataFrame) -> go.Figure:
    """Distribution of daily AQI per city over the last year (``city, date, aqi``)."""
    if daily_by_city.empty:
        return empty_figure()
    order = daily_by_city.groupby("city")["aqi"].median().sort_values(ascending=False).index
    fig = go.Figure()
    for city in order:
        values = daily_by_city.loc[daily_by_city["city"] == city, "aqi"]
        fig.add_box(y=values, name=city, marker_color=color_for_aqi(values.median()), boxmean=True)
    return fig.update_layout(title="Розкид добового AQI за останній рік", yaxis_title="AQI", showlegend=False,
                             **LAYOUT)


def heatmap_city_hour(matrix: pd.DataFrame) -> go.Figure:
    """Heatmap of mean AQI: city (rows) by hour (columns)."""
    return aqi_heatmap(matrix, [f"{h:02d}:00" for h in matrix.columns], list(matrix.index),
                       "Середній AQI за останній рік: місто × година", "Година доби", "Місто")
