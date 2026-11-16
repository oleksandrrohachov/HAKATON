"""Forecast charts: what WILL happen and how well the models did on data they had not seen.

Everything predicted is drawn in FORECAST_COLOR (purple, dashed); measured data is blue.
"""
import pandas as pd
import plotly.graph_objects as go

from charts.base import FORECAST_COLOR, HISTORY_COLOR, LAYOUT, NO_DATA_COLOR, aqi_colors, empty_figure

HISTORY_POINTS_SHOWN = 120
CORRIDOR_FILL = "rgba(123,97,255,0.18)"
TEST_FILL = "rgba(123,97,255,0.07)"
ACTUAL_COLOR = "#333333"
MODEL_LABELS = {
    "naive": "Як учора",
    "seasonal_naive": "Як тиждень тому",
    "holt_winters": "Хольта–Вінтерса (7 днів)",
    "train_mean": "Середнє з навчання",
    "last_year": "Як рік тому",
    "seasonal_hw": "Сезонна модель (52 тижні)",
}


def line_with_forecast(history: pd.DataFrame, forecast: pd.DataFrame,
                       title: str = "AQI: історія та прогноз") -> go.Figure:
    """Measured history (``date, aqi``) plus forecast (``date, yhat, lower, upper``) with its corridor."""
    history = history.dropna(subset=["aqi"]).tail(HISTORY_POINTS_SHOWN)
    if history.empty and forecast.empty:
        return empty_figure()
    fig = go.Figure()
    if not history.empty:
        fig.add_scatter(x=history["date"], y=history["aqi"], name="Виміряно", mode="lines+markers",
                        line={"color": HISTORY_COLOR}, marker={"color": aqi_colors(history["aqi"]), "size": 5})
    if not forecast.empty:
        fig.add_scatter(x=forecast["date"], y=forecast["upper"], mode="lines", line={"width": 0},
                        showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=forecast["date"], y=forecast["lower"], mode="lines", line={"width": 0},
                        fill="tonexty", fillcolor=CORRIDOR_FILL, name="Коридор невизначеності (95%)")
        fig.add_scatter(x=forecast["date"], y=forecast["yhat"], name="Прогноз", mode="lines+markers",
                        line={"color": FORECAST_COLOR, "dash": "dash", "width": 3},
                        marker={"color": FORECAST_COLOR, "size": 6})
    return fig.update_layout(title=title, xaxis_title="Дата", yaxis_title="AQI", hovermode="x unified", **LAYOUT)


def train_test_chart(train: pd.DataFrame, test: pd.DataFrame, model: str = "seasonal_hw") -> go.Figure:
    """80/20 check: training weeks (``week, aqi``), test weeks (``week, actual, <model>``)."""
    if train.empty or test.empty:
        return empty_figure()
    start, end = test["week"].min(), test["week"].max()
    fig = go.Figure()
    fig.add_vrect(x0=start, x1=end, fillcolor=TEST_FILL, line_width=0, layer="below")
    fig.add_scatter(x=train["week"], y=train["aqi"], name="Навчання: модель бачила (80%)", mode="lines",
                    line={"color": HISTORY_COLOR, "width": 1.5})
    fig.add_scatter(x=test["week"], y=test["actual"], name="Тест: реальність (20%)", mode="lines",
                    line={"color": ACTUAL_COLOR, "width": 2})
    fig.add_scatter(x=test["week"], y=test[model], name=f"Тест: прогноз — {MODEL_LABELS.get(model, model)}",
                    mode="lines", line={"color": FORECAST_COLOR, "dash": "dash", "width": 3})
    for x, text in ((train["week"].iloc[len(train) // 2], "НАВЧАННЯ 80%"), (start + (end - start) / 2, "ТЕСТ 20%")):
        fig.add_annotation(x=x, y=1.02, yref="paper", text=f"<b>{text}</b>", showarrow=False)
    return fig.update_layout(title="Перевірка 80/20: модель бачила лише навчальні 80% і прогнозувала тест",
                             xaxis_title="Тиждень", yaxis_title="AQI (середнє за тиждень)", hovermode="x unified",
                             legend={"orientation": "h", "y": -0.2}, **LAYOUT)


def predicted_vs_actual(backtest: pd.DataFrame, model: str = "holt_winters") -> go.Figure:
    """Scatter of forecast vs real AQI from a backtest table (perfect forecast = diagonal)."""
    if backtest.empty:
        return empty_figure()
    top = float(max(backtest["actual"].max(), backtest[model].max()))
    fig = go.Figure()
    fig.add_scatter(x=[0, top], y=[0, top], mode="lines", name="Ідеальний прогноз",
                    line={"color": NO_DATA_COLOR, "dash": "dash"})
    fig.add_scatter(x=backtest["actual"], y=backtest[model], mode="markers", name="Прогноз",
                    marker={"color": aqi_colors(backtest["actual"]), "size": 7, "opacity": 0.8})
    return fig.update_layout(title="Прогноз проти реальності", xaxis_title="Реальний AQI",
                             yaxis_title="Прогнозований AQI", **LAYOUT)


def model_comparison(errors: pd.DataFrame, title: str = "Помилка моделей (MAE, менше — краще)") -> go.Figure:
    """Bars of MAE per model (``model, mae``); the lower the better."""
    if errors.empty:
        return empty_figure()
    labels = [MODEL_LABELS.get(model, model) for model in errors["model"]]
    colors = [FORECAST_COLOR if model in ("holt_winters", "seasonal_hw") else NO_DATA_COLOR
              for model in errors["model"]]
    fig = go.Figure(go.Bar(x=labels, y=errors["mae"], marker_color=colors,
                           text=[f"{v:.1f}" for v in errors["mae"]], textposition="outside"))
    return fig.update_layout(title=title, yaxis_title="MAE, одиниць AQI", **LAYOUT)
