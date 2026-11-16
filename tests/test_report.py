import plotly.graph_objects as go

import report


def test_build_summary_contains_all_sections(hourly_df):
    text = report.build_summary("kyiv", {"kyiv": hourly_df, "lviv": hourly_df})
    for expected in ("Київ", "А. АНАЛІТИКА", "1. Зараз", "2. Закономірності", "3. Порівняння міст",
                     "Б. ПРОГНОЗ", "4. Короткостроковий", "5. Перевірка", "6. Сезонна модель", "SaveEcoBot"):
        assert expected in text


def test_labelled_frames_use_ukrainian_names(hourly_df):
    assert list(report.labelled_frames({"kyiv": hourly_df, "unknown": hourly_df})) == ["Київ", "Unknown"]


def test_figures_are_grouped_into_analytics_forecast_cities(yearly_df):
    figures = report.build_figures("kyiv", {"kyiv": yearly_df, "lviv": yearly_df})
    groups = [name.split("_")[1] for name in figures]
    assert groups == sorted(groups, key=["analytics", "forecast", "cities"].index)
    assert all(isinstance(fig, go.Figure) and len(fig.data) > 0 for fig in figures.values())
