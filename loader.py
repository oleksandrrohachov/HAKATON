"""Reading and cleaning SaveEcoBot hourly CSV files.

Each file in ``data/`` is named ``<city>.csv`` and has a *long* format:
``phenomenon,value,logged_at`` (pm25, aqi, temperature, ...). We keep only
``pm25`` and ``aqi`` and pivot them into one row per hour.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from cities import DATA_DIR, city_label  # noqa: F401  (city_label is re-exported for app/report)

REQUIRED_COLUMNS = {"phenomenon", "value", "logged_at"}
KEPT_PHENOMENA = ("pm25", "aqi")

# 500.4 is the top of the EPA scale; larger PM2.5 values (e.g. the 999.9 sensor
# cap in lviv.csv) are treated as sensor errors, stricter than the plain "> 1000".
PM25_MAX = 500.4
AQI_MAX = 500

# US EPA PM2.5 breakpoints (2012 table, it matches the AQI in the files best):
# (pm25_low, pm25_high, aqi_low, aqi_high)
EPA_PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 500.4, 301, 500),
]


def list_cities() -> list[str]:
    """Return sorted city names, i.e. the stems of ``data/*.csv``."""
    return sorted(path.stem for path in DATA_DIR.glob("*.csv"))


def pm25_to_aqi(pm25: pd.Series) -> pd.Series:
    """Convert PM2.5 (ug/m3) to US EPA AQI with linear interpolation."""
    xs = [x for low, high, _, _ in EPA_PM25_BREAKPOINTS for x in (low, high)]
    ys = [y for _, _, low, high in EPA_PM25_BREAKPOINTS for y in (low, high)]
    # np.interp is piecewise-linear between all knots; NaN stays NaN.
    return pd.Series(np.interp(pm25, xs, ys), index=pm25.index).where(pm25.notna())


def _find_city_file(city: str) -> Path:
    """Find ``data/<city>.csv`` ignoring case, or raise a helpful error."""
    for path in DATA_DIR.glob("*.csv"):
        if path.stem.lower() == city.strip().lower():
            return path
    available = ", ".join(list_cities()) or "немає файлів"
    raise FileNotFoundError(f"Місто '{city}' не знайдено. Доступні міста: {available}.")


def _read_long_csv(path: Path) -> pd.DataFrame:
    """Read a long-format CSV and keep only valid pm25/aqi rows."""
    try:
        raw = pd.read_csv(path)
    except (pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Файл {path.name} пошкоджений: {exc}") from exc
    if not REQUIRED_COLUMNS.issubset(raw.columns):
        raise ValueError(
            f"У файлі {path.name} очікуються колонки {sorted(REQUIRED_COLUMNS)}, "
            f"а є {list(raw.columns)}."
        )
    raw = raw[raw["phenomenon"].isin(KEPT_PHENOMENA)]
    return pd.DataFrame(
        {
            "phenomenon": raw["phenomenon"],
            # Non-numeric -> NaN, nothing is invented.
            "value": pd.to_numeric(raw["value"], errors="coerce"),
            # Floor to the hour so rare off-hour timestamps join their hour.
            "time": pd.to_datetime(raw["logged_at"], errors="coerce").dt.floor("h"),
        }
    ).dropna(subset=["time"])


def _mask_invalid(df: pd.DataFrame) -> pd.DataFrame:
    """Set negative values and outliers to NaN (rows are kept)."""
    for column, upper in (("pm25", PM25_MAX), ("aqi", AQI_MAX)):
        bad = (df[column] < 0) | (df[column] > upper)
        df[column] = df[column].mask(bad)
    return df


def load_city(city: str) -> pd.DataFrame:
    """Load one city as an hourly table with columns ``time, pm25, aqi``.

    ``time`` is datetime64, sorted, without duplicates (duplicates are averaged).
    Gaps stay as missing rows/NaN. If the file has no AQI at all, it is computed
    from PM2.5 with the US EPA table.
    """
    long_df = _read_long_csv(_find_city_file(city))
    wide = (
        long_df.pivot_table(index="time", columns="phenomenon", values="value", aggfunc="mean")
        .reindex(columns=list(KEPT_PHENOMENA))
        .reset_index()
        .sort_values("time", ignore_index=True)
    )
    wide.columns.name = None
    wide = _mask_invalid(wide)
    if wide["aqi"].isna().all():
        wide["aqi"] = pm25_to_aqi(wide["pm25"])
    return wide
