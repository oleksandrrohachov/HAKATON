"""Streamlit entry point "Чисте небо": page frame, sidebar and tabs (content lives in ``ui/``)."""
import streamlit as st

import loader
from ui import analytics_tab, cities_tab, data, forecast_tab, glossary_tab, now_card

DATA_SOURCE_URL = "https://www.saveecobot.com/"
DEFAULT_CITY = "kyiv"
FORECAST_DAYS_RANGE = (3, 14, 7)  # min, max, default
MIN_HOURS_FOR_ANALYSIS = 7 * 24
TAB_NAMES = ["📊 Аналітика: що було", "🔮 Прогноз: що буде", "🏙️ Порівняння міст", "📖 Словник"]


def render_sidebar(cities: list[str]) -> tuple[str, int]:
    """Sidebar controls: city and forecast length."""
    st.sidebar.header("Налаштування")
    index = cities.index(DEFAULT_CITY) if DEFAULT_CITY in cities else 0
    city = st.sidebar.selectbox("Місто", cities, index=index, format_func=loader.city_label)
    low, high, default = FORECAST_DAYS_RANGE
    days = st.sidebar.slider("Днів короткострокового прогнозу", low, high, default)
    return city, days


def render_footer() -> None:
    """License attribution required by CC BY 4.0."""
    st.divider()
    st.markdown(f"Дані: [SaveEcoBot]({DATA_SOURCE_URL}) (CC BY 4.0)")


def main() -> None:
    """Page entry point."""
    st.set_page_config(page_title="Чисте небо", page_icon="🌤️", layout="wide")
    st.title("🌤️ Чисте небо")
    st.caption("Якість повітря в містах України: аналітика минулого і прогноз майбутнього.")
    cities = loader.list_cities()
    if not cities:
        st.error("У папці data/ немає CSV-файлів із даними.")
        return render_footer()
    city, days = render_sidebar(cities)
    try:
        df = data.city_data(city)
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Не вдалося завантажити дані: {exc}")
        return render_footer()
    if df["aqi"].notna().sum() < MIN_HOURS_FOR_ANALYSIS:
        st.warning("Для цього міста замало даних, щоб показати аналіз.")
        return render_footer()
    now_card.render(df)
    analytics, prediction, comparison, terms = st.tabs(TAB_NAMES)
    with analytics:
        analytics_tab.render(city, df)
    with prediction:
        forecast_tab.render(city, days)
    with comparison:
        cities_tab.render(cities)
    with terms:
        glossary_tab.render()
    render_footer()


main()
