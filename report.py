"""Static report: PNG charts and a text summary for one city plus the city comparison.

Usage: ``python report.py [city]`` (default: kyiv). Output goes to ``outputs/``.
File names start with ``analytics_`` (what has happened), ``forecast_`` (what will happen)
or ``cities_`` (comparison), so the two parts are never mixed up.
"""
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

import analysis
import charts
import forecast
import insights
import loader
import seasonal

OUTPUT_DIR = Path(__file__).parent / "outputs"
DEFAULT_CITY = "kyiv"
SHORT_FORECAST_DAYS = 7
OUTLOOK_ROW_STEP = 4      # show every 4th week of the half-year forecast in the text report
IMAGE_SIZE = (1100, 600)  # width, height of PNG files
IMAGE_SCALE = 1.5
DISCLAIMER = (
    "Прогноз — статистична оцінка за історією вимірювань, а не медична порада. "
    "Дані: SaveEcoBot (CC BY 4.0), https://www.saveecobot.com/"
)


def labelled_frames(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Same data keyed by the Ukrainian city names (for chart and table labels)."""
    return {loader.city_label(name): df for name, df in dfs.items()}


def analytics_figures(df: pd.DataFrame) -> dict[str, go.Figure]:
    """Charts of measured data only."""
    return {
        "01_analytics_weekly_trend": charts.weekly_trend(analysis.weekly(df)),
        "02_analytics_hour_weekday": charts.heatmap_hour_weekday(analysis.hour_weekday_matrix(df)),
        "03_analytics_hourly_profile": charts.hourly_profile_chart(analysis.hourly_profile(df)),
        "04_analytics_weekday_profile": charts.weekday_profile_chart(analysis.weekday_profile(df)),
        "05_analytics_month_year": charts.heatmap_month_year(analysis.month_year_matrix(df)),
        "06_analytics_category_share": charts.category_share_chart(insights.category_share_by_year(df)),
        "07_analytics_decomposition": charts.decomposition_chart(forecast.decompose_daily(df)),
    }


def forecast_figures(df: pd.DataFrame) -> dict[str, go.Figure]:
    """Forecasts and their checks (short-term rolling check and the 80/20 seasonal check)."""
    backtest = forecast.rolling_backtest(df, SHORT_FORECAST_DAYS)
    train, test = seasonal.seasonal_split(df)
    weekly_history = seasonal.weekly_series(df).rename_axis("date").reset_index()
    return {
        "08_forecast_short_7_days": charts.line_with_forecast(
            analysis.daily(df), forecast.forecast_aqi(df, SHORT_FORECAST_DAYS), title="Прогноз AQI на 7 днів"),
        "09_forecast_short_models": charts.model_comparison(
            forecast.model_errors(backtest), title="Помилка 7-денних моделей (MAE)"),
        "10_forecast_short_vs_actual": charts.predicted_vs_actual(backtest, "holt_winters"),
        "11_forecast_train_test_80_20": charts.train_test_chart(train, test),
        "12_forecast_seasonal_models": charts.model_comparison(
            forecast.model_errors(test), title="Помилка на тесті 20% (MAE)"),
        "13_forecast_seasonal_26_weeks": charts.line_with_forecast(
            weekly_history, seasonal.seasonal_forecast(df), title="Тижневий AQI: історія та прогноз на 26 тижнів"),
    }


def cities_figures(dfs: dict[str, pd.DataFrame]) -> dict[str, go.Figure]:
    """Comparison of all cities."""
    labelled = labelled_frames(dfs)
    return {
        "14_cities_compare": charts.compare_cities(labelled),
        "15_cities_boxplot": charts.city_boxplot(analysis.daily_by_city(labelled)),
        "16_cities_hour": charts.heatmap_city_hour(analysis.city_hour_matrix(labelled)),
    }


def build_figures(city: str, dfs: dict[str, pd.DataFrame]) -> dict[str, go.Figure]:
    """All report charts, keyed by file name (without extension), in display order."""
    return {**analytics_figures(dfs[city]), **forecast_figures(dfs[city]), **cities_figures(dfs)}


def save_figures(figures: dict[str, go.Figure], plots_dir: Path) -> None:
    """Replace the PNG files in ``plots_dir`` (one browser session for all files)."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    for old in plots_dir.glob("*.png"):
        old.unlink()
    width, height = IMAGE_SIZE
    for fig in figures.values():
        fig.update_layout(width=width, height=fig.layout.height or height)
    pio.write_images(fig=list(figures.values()), file=[plots_dir / f"{name}.png" for name in figures],
                     scale=IMAGE_SCALE)


def _errors_by_city(dfs: dict[str, pd.DataFrame], make_backtest: Callable, models: list[str]) -> pd.DataFrame:
    """MAE of every model for every city where the check is possible (rows = city)."""
    rows = {}
    for city, df in dfs.items():
        try:
            rows[loader.city_label(city)] = forecast.model_errors(make_backtest(df)).set_index("model")["mae"]
        except ValueError:
            continue
    return pd.DataFrame(rows).T.reindex(columns=models)


def short_term_errors(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """MAE of the 7-day models on the last 8 weeks, per city."""
    return _errors_by_city(dfs, lambda df: forecast.rolling_backtest(df, SHORT_FORECAST_DAYS), forecast.MODELS)


def seasonal_errors(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """MAE of the seasonal models on the 20% test part, per city."""
    return _errors_by_city(dfs, lambda df: seasonal.seasonal_split(df)[1], seasonal.SEASONAL_MODELS)


def _section(title: str, lines: list[str]) -> str:
    """Text block with an underlined title."""
    return "\n".join([title, "-" * len(title), *lines, ""])


def _wins(errors: pd.DataFrame, model: str) -> str:
    """'model is best in N of M cities' sentence."""
    wins = int((errors.idxmin(axis=1) == model).sum()) if not errors.empty else 0
    return f"{charts.MODEL_LABELS[model]} найкраща у {wins} з {len(errors)} міст."


def _analytics_text(city: str, dfs: dict[str, pd.DataFrame]) -> list[str]:
    df, info = dfs[city], analysis.summary(dfs[city])
    stats = analysis.city_stats(labelled_frames(dfs)).sort_values("mean_aqi")
    return [
        "А. АНАЛІТИКА — ЩО БУЛО (лише виміряні дані)", "",
        _section("1. Зараз", [
            f"Останній AQI: {info['last_aqi']:.0f} ({insights.aqi_category(info['last_aqi'])}) "
            f"на {info['last_time']:%Y-%m-%d %H:%M}",
            f"Порада: {insights.advice(info['last_aqi'])}",
            f"Середній AQI за 30 днів: {info['mean_aqi_30d']:.1f}; найгірший день: "
            f"{info['worst_day']:%Y-%m-%d} (AQI {info['worst_day_aqi']:.0f})",
        ]),
        _section("2. Закономірності", [f"- {text}" for text in insights.key_findings(df)]),
        _section("3. Порівняння міст за останній рік", [stats.round(1).to_string(index=False)]),
    ]


def _seasonal_text(df: pd.DataFrame) -> list[str]:
    try:
        train, test = seasonal.seasonal_split(df)
    except ValueError as exc:
        return [str(exc)]
    outlook = seasonal.seasonal_forecast(df).iloc[::OUTLOOK_ROW_STEP].round({"yhat": 1, "lower": 1, "upper": 1})
    return [
        f"Навчання (80%): {train['week'].min():%Y-%m-%d} — {train['week'].max():%Y-%m-%d}, {len(train)} тижнів",
        f"Тест (20%):     {test['week'].min():%Y-%m-%d} — {test['week'].max():%Y-%m-%d}, {len(test)} тижнів",
        "", "Прогноз на 26 тижнів (кожен 4-й тиждень, AQI):",
        outlook.assign(date=outlook["date"].dt.strftime("%Y-%m-%d")).to_string(index=False),
    ]


def _forecast_text(city: str, dfs: dict[str, pd.DataFrame]) -> list[str]:
    df = dfs[city]
    short = forecast.forecast_aqi(df, SHORT_FORECAST_DAYS).round({"yhat": 1, "lower": 1, "upper": 1})
    short_errors, long_errors = short_term_errors(dfs), seasonal_errors(dfs)
    return [
        "Б. ПРОГНОЗ — ЩО БУДЕ (модель вчиться на минулому)", "",
        _section(f"4. Короткостроковий прогноз на {SHORT_FORECAST_DAYS} днів (коридор 95%)",
                 [short.assign(date=short["date"].dt.strftime("%Y-%m-%d")).to_string(index=False)]),
        _section("5. Перевірка 7-денного прогнозу на 8 останніх тижнях (MAE, менше — краще)", [
            short_errors.round(2).rename(columns=charts.MODEL_LABELS).to_string(),
            _wins(short_errors, "holt_winters") + " Добовий AQI інерційний, тому «як учора» — сильний еталон.",
        ]),
        _section("6. Сезонна модель: 80% навчання / 20% тест", [
            *_seasonal_text(df), "",
            "MAE на тесті 20% по містах:",
            long_errors.round(2).rename(columns=charts.MODEL_LABELS).to_string(),
            _wins(long_errors, "seasonal_hw"),
        ]),
    ]


def build_summary(city: str, dfs: dict[str, pd.DataFrame]) -> str:
    """Text report: analytics (what has happened) and forecast (what will happen) apart."""
    df = dfs[city]
    header = [
        f"ЧИСТЕ НЕБО: звіт по місту {loader.city_label(city)}",
        f"Сформовано: {datetime.now():%Y-%m-%d %H:%M}. Період даних: {df['time'].min():%Y-%m-%d} — "
        f"{df['time'].max():%Y-%m-%d}, годинних записів: {len(df):,}",
        DISCLAIMER, "",
    ]
    return "\n".join([*header, *_analytics_text(city, dfs), *_forecast_text(city, dfs)])


def main(city: str = DEFAULT_CITY) -> None:
    """Build all charts and the summary for ``city`` into ``outputs/``."""
    dfs = {name: loader.load_city(name) for name in loader.list_cities()}
    city = city.lower()
    if city not in dfs:
        raise SystemExit(f"Невідоме місто '{city}'. Доступні: {', '.join(dfs)}")
    save_figures(build_figures(city, dfs), OUTPUT_DIR / "plots")
    reports_dir = OUTPUT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "summary.txt").write_text(build_summary(city, dfs), encoding="utf-8")
    analysis.city_stats(labelled_frames(dfs)).round(2).to_csv(reports_dir / "city_stats.csv", index=False)
    print(f"Готово: графіки в {OUTPUT_DIR / 'plots'}, звіт у {reports_dir / 'summary.txt'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(*sys.argv[1:2])
