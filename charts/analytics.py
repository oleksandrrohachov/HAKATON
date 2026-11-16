"""Analytics charts: what HAS happened (measured data only, nothing predicted)."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from charts.base import (
    ACCENT_COLOR, AQI_COLORS, HISTORY_COLOR, LAYOUT, MONTH_LABELS, WEEKDAY_LABELS, aqi_colors, aqi_heatmap,
    empty_figure,
)
from insights import AQI_CATEGORIES

SMOOTHING_WEEKS = 4


def weekly_trend(weekly: pd.DataFrame) -> go.Figure:
    """Weekly mean AQI with a 4-week moving average (``week, aqi``)."""
    weekly = weekly.dropna(subset=["aqi"])
    if weekly.empty:
        return empty_figure()
    smooth = weekly["aqi"].rolling(SMOOTHING_WEEKS, min_periods=1).mean()
    fig = go.Figure()
    fig.add_scatter(x=weekly["week"], y=weekly["aqi"], name="AQI за тиждень", mode="lines",
                    line={"color": HISTORY_COLOR, "width": 1}, opacity=0.5)
    fig.add_scatter(x=weekly["week"], y=smooth, name=f"Ковзне середнє ({SMOOTHING_WEEKS} тижні)", mode="lines",
                    line={"color": ACCENT_COLOR, "width": 3})
    return fig.update_layout(title="Тижневий AQI за весь період", xaxis_title="Тиждень", yaxis_title="AQI",
                             hovermode="x unified", **LAYOUT)


def hourly_profile_chart(hourly: pd.DataFrame) -> go.Figure:
    """Mean AQI by hour of day (``hour, aqi``)."""
    hourly = hourly.dropna(subset=["aqi"])
    if hourly.empty:
        return empty_figure()
    fig = go.Figure(go.Scatter(x=[f"{h:02d}:00" for h in hourly["hour"]], y=hourly["aqi"], mode="lines+markers",
                               line={"color": HISTORY_COLOR}, marker={"color": aqi_colors(hourly["aqi"]), "size": 8}))
    return fig.update_layout(title="Середній AQI за годинами доби", xaxis_title="Година", yaxis_title="AQI",
                             **LAYOUT)


def weekday_profile_chart(weekday: pd.DataFrame) -> go.Figure:
    """Mean AQI by weekday (``weekday, aqi``)."""
    weekday = weekday.dropna(subset=["aqi"])
    if weekday.empty:
        return empty_figure()
    fig = go.Figure(go.Bar(x=[WEEKDAY_LABELS[d] for d in weekday["weekday"]], y=weekday["aqi"],
                           marker_color=aqi_colors(weekday["aqi"]), text=[f"{v:.0f}" for v in weekday["aqi"]],
                           textposition="outside"))
    return fig.update_layout(title="Середній AQI за днями тижня", xaxis_title="День тижня", yaxis_title="AQI",
                             **LAYOUT)


def heatmap_hour_weekday(matrix: pd.DataFrame) -> go.Figure:
    """Heatmap of mean AQI: weekday (rows) by hour (columns)."""
    return aqi_heatmap(matrix, [f"{h:02d}:00" for h in matrix.columns], [WEEKDAY_LABELS[d] for d in matrix.index],
                       "Середній AQI: година × день тижня", "Година доби", "День тижня")


def heatmap_month_year(matrix: pd.DataFrame) -> go.Figure:
    """Heatmap of mean AQI: year (rows) by month (columns)."""
    return aqi_heatmap(matrix, [MONTH_LABELS[m - 1] for m in matrix.columns], [str(y) for y in matrix.index],
                       "Середній AQI: рік × місяць (опалювальний сезон)", "Місяць", "Рік")


def category_share_chart(shares: pd.DataFrame) -> go.Figure:
    """Stacked bars: % of days in each AQI category per year (rows = year, columns = band)."""
    if shares.empty:
        return empty_figure()
    fig = go.Figure()
    for band, name in enumerate(AQI_CATEGORIES):
        if band in shares.columns and shares[band].sum() > 0:
            fig.add_bar(x=[str(y) for y in shares.index], y=shares[band], name=name, marker_color=AQI_COLORS[band])
    return fig.update_layout(barmode="stack", title="Частка днів за категоріями якості повітря",
                             xaxis_title="Рік", yaxis_title="% днів", **LAYOUT)


def decomposition_chart(parts: pd.DataFrame) -> go.Figure:
    """Four stacked panels: observed, trend, seasonal, residual (``date, observed, trend, seasonal, resid``)."""
    if parts.empty:
        return empty_figure()
    titles = {"observed": "Спостереження", "trend": "Тренд", "seasonal": "Сезонність", "resid": "Залишок"}
    fig = make_subplots(rows=len(titles), cols=1, shared_xaxes=True, subplot_titles=list(titles.values()),
                        vertical_spacing=0.06)
    for row, column in enumerate(titles, start=1):
        fig.add_scatter(x=parts["date"], y=parts[column], mode="lines", showlegend=False,
                        line={"color": ACCENT_COLOR if column == "trend" else HISTORY_COLOR, "width": 1.5},
                        row=row, col=1)
    return fig.update_layout(title="Розклад добового AQI: тренд і сезонність", height=700, **LAYOUT)
