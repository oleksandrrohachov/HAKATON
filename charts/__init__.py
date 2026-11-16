"""Plotly charts, split by topic. Every chart takes its colours from ``color_for_aqi``.

- ``charts.base``: colour code, layout, shared heatmap;
- ``charts.analytics``: what has happened (measured data);
- ``charts.prediction``: forecasts and their checks;
- ``charts.cities``: city comparison.
"""
from charts.analytics import (
    category_share_chart, decomposition_chart, heatmap_hour_weekday, heatmap_month_year, hourly_profile_chart,
    weekday_profile_chart, weekly_trend,
)
from charts.base import AQI_COLORS, NO_DATA_COLOR, color_for_aqi
from charts.cities import city_boxplot, compare_cities, heatmap_city_hour
from charts.prediction import MODEL_LABELS, line_with_forecast, model_comparison, predicted_vs_actual, train_test_chart
