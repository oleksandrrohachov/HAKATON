"""Human-readable texts: AQI categories, advice, best hour, year-over-year change."""
import bisect
import math

import pandas as pd

from analysis import (
    MIN_DAYS_PER_YEAR, STATS_WINDOW_DAYS, UNHEALTHY_AQI, daily, hourly_profile, last_days, monthly,
    weekday_profile, yearly,
)

# Upper AQI bound (inclusive) of each band, US EPA style. The last band is open-ended.
AQI_UPPER_BOUNDS = [50, 100, 150, 200, 300]
AQI_CATEGORIES = [
    "Добре",
    "Помірно",
    "Шкідливо для чутливих",
    "Шкідливо",
    "Дуже шкідливо",
    "Небезпечно",
]
AQI_ADVICE = [
    "Повітря чисте. Гуляйте, провітрюйте квартиру, займайтеся спортом надворі.",
    "Прийнятно для більшості. Дуже чутливим людям краще скоротити тривале навантаження надворі.",
    "Закрийте вікна, якщо ви чутливі до забруднення. Дітям, літнім людям і астматикам краще менше бувати надворі.",
    "Закрийте вікна. Обмежте прогулянки та спорт надворі, на вулиці бажана маска FFP2.",
    "Закрийте вікна, увімкніть очисник повітря. Не виходьте надовго на вулицю.",
    "Небезпечне повітря. Залишайтеся вдома з закритими вікнами, на вулиці тільки в респіраторі.",
]
NO_DATA = "Немає даних"
MONTH_NAMES = [
    "січні", "лютому", "березні", "квітні", "травні", "червні",
    "липні", "серпні", "вересні", "жовтні", "листопаді", "грудні",
]
SAME_LEVEL_PERCENT = 1.0  # changes below this are reported as "about the same"
WEEKEND_DAYS = (5, 6)     # Saturday, Sunday
MONTH_NAMES_NOMINATIVE = [
    "січень", "лютий", "березень", "квітень", "травень", "червень",
    "липень", "серпень", "вересень", "жовтень", "листопад", "грудень",
]


def aqi_band(aqi: float) -> int | None:
    """Index (0-5) of the AQI band, or ``None`` for a missing value."""
    if aqi is None or math.isnan(aqi):
        return None
    return bisect.bisect_left(AQI_UPPER_BOUNDS, aqi)


def aqi_category(aqi: float) -> str:
    """Category name for an AQI value, e.g. 42 -> "Добре"."""
    band = aqi_band(aqi)
    return NO_DATA if band is None else AQI_CATEGORIES[band]


def advice(aqi: float) -> str:
    """Short advice for a person at the given AQI."""
    band = aqi_band(aqi)
    return NO_DATA if band is None else AQI_ADVICE[band]


def best_hour(hourly: pd.DataFrame) -> str:
    """Hour of day with the lowest AQI, e.g. "06:00" (input: ``hourly_profile``)."""
    column = "aqi" if hourly["aqi"].notna().any() else "pm25"
    if hourly.empty or hourly[column].notna().sum() == 0:
        return NO_DATA
    hour = int(hourly.loc[hourly[column].idxmin(), "hour"])
    return f"{hour:02d}:00"


def compare_to_last_year(df: pd.DataFrame, month: int) -> str:
    """Compare the latest available ``month`` with the same month a year earlier.

    Example: "на 12% чистіше, ніж торік". The latest month may be incomplete
    (e.g. the current one); we still compare month means, which is good enough.
    """
    table = monthly(df).dropna(subset=["aqi"])
    table = table[table["month"] == month].set_index("year")["aqi"]
    if table.empty:
        return f"Даних у {MONTH_NAMES[month - 1]} немає."
    year = int(table.index.max())
    if year - 1 not in table.index or table[year - 1] <= 0:
        return f"Даних у {MONTH_NAMES[month - 1]} минулого року немає, порівняти неможливо."
    change = (table[year] - table[year - 1]) / table[year - 1] * 100
    if abs(change) < SAME_LEVEL_PERCENT:
        return "приблизно так само, як торік"
    word = "чистіше" if change < 0 else "брудніше"
    return f"на {abs(change):.0f}% {word}, ніж торік"


def category_share_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Share (%) of days in each AQI category per year.

    Rows = year, columns = category index 0-5 (see ``AQI_CATEGORIES``).
    """
    days = daily(df).dropna(subset=["aqi"])
    bands = days["aqi"].map(aqi_band)
    counts = pd.crosstab(days["date"].dt.year, bands).reindex(columns=range(len(AQI_CATEGORIES)), fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0).mul(100).rename_axis(index="year", columns="band")


def _percent_diff(new: float, old: float) -> float:
    """Relative change of ``new`` versus ``old`` in percent."""
    return (new - old) / old * 100


def _hour_finding(df: pd.DataFrame) -> str | None:
    profile = hourly_profile(df).dropna(subset=["aqi"])
    if profile.empty:
        return None
    best, worst = profile.loc[profile["aqi"].idxmin()], profile.loc[profile["aqi"].idxmax()]
    return (f"Найчистіше повітря о {int(best['hour']):02d}:00 (AQI {best['aqi']:.0f}), "
            f"найбрудніше о {int(worst['hour']):02d}:00 (AQI {worst['aqi']:.0f}).")


def _weekend_finding(df: pd.DataFrame) -> str | None:
    profile = weekday_profile(df).dropna(subset=["aqi"]).set_index("weekday")["aqi"]
    weekend, weekdays = profile[profile.index.isin(WEEKEND_DAYS)], profile[~profile.index.isin(WEEKEND_DAYS)]
    if weekend.empty or weekdays.empty:
        return None
    change = _percent_diff(weekend.mean(), weekdays.mean())
    if abs(change) < SAME_LEVEL_PERCENT:
        return "У вихідні повітря таке саме, як у будні."
    return f"У вихідні повітря на {abs(change):.0f}% {'чистіше' if change < 0 else 'брудніше'}, ніж у будні."


def _season_finding(df: pd.DataFrame) -> str | None:
    by_month = monthly(df).dropna(subset=["aqi"]).groupby("month")["aqi"].mean()
    if len(by_month) < 12:
        return None
    worst, best = by_month.idxmax(), by_month.idxmin()
    return (f"Найгірший місяць — {MONTH_NAMES_NOMINATIVE[worst - 1]} (AQI {by_month[worst]:.0f}), "
            f"найчистіший — {MONTH_NAMES_NOMINATIVE[best - 1]} (AQI {by_month[best]:.0f}): "
            f"різниця в {by_month[worst] / by_month[best]:.1f} раза.")


def _year_finding(df: pd.DataFrame) -> str | None:
    years = yearly(df)
    years = years[years["days"] >= MIN_DAYS_PER_YEAR].set_index("year")["aqi"]
    if len(years) < 2:
        return None
    last, previous = years.index[-1], years.index[-2]
    change = _percent_diff(years[last], years[previous])
    word = "чистіше" if change < 0 else "брудніше"
    return f"{last} рік у середньому на {abs(change):.0f}% {word}, ніж {previous}."


def _unhealthy_finding(df: pd.DataFrame) -> str | None:
    recent = last_days(df, STATS_WINDOW_DAYS).dropna(subset=["aqi"])
    if recent.empty:
        return None
    share = (recent["aqi"] > UNHEALTHY_AQI).mean() * 100
    return f"За останній рік {share:.1f}% годин повітря було шкідливим для чутливих людей (AQI > {UNHEALTHY_AQI})."


def key_findings(df: pd.DataFrame) -> list[str]:
    """Plain-language conclusions about one city (skips what cannot be computed)."""
    findings = [f(df) for f in (_hour_finding, _weekend_finding, _season_finding, _year_finding,
                                _unhealthy_finding)]
    return [text for text in findings if text]
