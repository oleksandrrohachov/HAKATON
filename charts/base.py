"""Shared chart building blocks: the AQI colour code, layout, empty figure, heatmap."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from insights import AQI_UPPER_BOUNDS, aqi_band

# One colour per AQI band (green -> red -> maroon), same order as insights.AQI_CATEGORIES.
AQI_COLORS = ["#2e9e5b", "#e3c21f", "#f08a24", "#d64545", "#8e3a9d", "#7a1f2e"]
NO_DATA_COLOR = "#9aa0a6"
HISTORY_COLOR = "#4c78a8"     # measured data (analytics)
FORECAST_COLOR = "#7b61ff"    # anything predicted (forecast) is purple
ACCENT_COLOR = "#e05c5c"
HEATMAP_STEPS = 24            # number of colour stops in a heatmap gradient
WEEKDAY_LABELS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]
MONTH_LABELS = ["Січ", "Лют", "Бер", "Кві", "Тра", "Чер", "Лип", "Сер", "Вер", "Жов", "Лис", "Гру"]
EMPTY_TEXT = "Немає даних для відображення"
LAYOUT = {"template": "plotly_white", "margin": {"l": 40, "r": 20, "t": 50, "b": 40}}


def color_for_aqi(aqi: float) -> str:
    """Hex colour for an AQI value (grey when missing)."""
    band = aqi_band(aqi)
    return NO_DATA_COLOR if band is None else AQI_COLORS[band]


def aqi_colors(values: pd.Series) -> list[str]:
    """AQI colour of every value (used for markers and bars)."""
    return [color_for_aqi(value) for value in values]


def empty_figure(text: str = EMPTY_TEXT) -> go.Figure:
    """Empty figure with a centered caption (used instead of raising on empty data)."""
    fig = go.Figure()
    fig.add_annotation(text=text, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font={"size": 16})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig.update_layout(**LAYOUT)


def gradient_color(aqi: float) -> str:
    """Smooth colour for an AQI value: blends the band colours around the band centres."""
    centres = [(low + high) / 2 for low, high in zip([0, *AQI_UPPER_BOUNDS], AQI_UPPER_BOUNDS)]
    centres.append(AQI_UPPER_BOUNDS[-1] + 50)  # open-ended last band
    channels = [np.interp(aqi, centres, [int(color[i:i + 2], 16) for color in AQI_COLORS]) for i in (1, 3, 5)]
    return "rgb({:.0f},{:.0f},{:.0f})".format(*channels)


def aqi_colorscale(low: float, high: float) -> list[list]:
    """Plotly colorscale that maps ``low..high`` to the AQI gradient."""
    values = np.linspace(low, high, HEATMAP_STEPS)
    return [[i / (HEATMAP_STEPS - 1), gradient_color(v)] for i, v in enumerate(values)]


def aqi_heatmap(matrix: pd.DataFrame, x_labels: list[str], y_labels: list[str], title: str,
                xaxis: str, yaxis: str) -> go.Figure:
    """Shared AQI heatmap (rows = y, columns = x); colour range follows the data."""
    if matrix.empty or matrix.isna().all().all():
        return empty_figure()
    low, high = float(np.nanmin(matrix.to_numpy())), float(np.nanmax(matrix.to_numpy()))
    high = max(high, low + 1)  # avoid a zero-width colour range
    fig = go.Figure(go.Heatmap(
        z=matrix.to_numpy(), x=x_labels, y=y_labels, colorscale=aqi_colorscale(low, high),
        zmin=low, zmax=high, colorbar={"title": "AQI"},
        hovertemplate="%{y}, %{x}<br>AQI: %{z:.0f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    return fig.update_layout(title=title, xaxis_title=xaxis, yaxis_title=yaxis, **LAYOUT)
