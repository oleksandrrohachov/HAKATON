"""Pandas aggregations over the hourly table ``time, pm25, aqi``."""
import pandas as pd

METRICS = ["pm25", "aqi"]
SUMMARY_WINDOW_DAYS = 30
STATS_WINDOW_DAYS = 365   # city comparison tables look at the last year of each city
UNHEALTHY_AQI = 100       # above this the air is unhealthy for sensitive people
MIN_DAYS_PER_YEAR = 360   # only (almost) full calendar years are compared: seasons differ a lot


def last_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Rows of the last ``days`` days; the window ends at the latest measurement."""
    if df.empty:
        return df
    return df[df["time"] > df["time"].max() - pd.Timedelta(days=days)]


def daily(df: pd.DataFrame) -> pd.DataFrame:
    """Daily means: ``date`` (midnight timestamp), ``pm25``, ``aqi``."""
    grouped = df.groupby(df["time"].dt.normalize())[METRICS].mean()
    return grouped.rename_axis("date").reset_index()


def hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Mean pm25/aqi by hour of day: ``hour`` 0-23, ``pm25``, ``aqi``."""
    return df.groupby(df["time"].dt.hour)[METRICS].mean().rename_axis("hour").reset_index()


def weekday_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Mean pm25/aqi by weekday: ``weekday`` 0=Mon..6=Sun, ``pm25``, ``aqi``."""
    return (
        df.groupby(df["time"].dt.day_of_week)[METRICS].mean().rename_axis("weekday").reset_index()
    )


def hour_weekday_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Mean AQI: rows = weekday (0-6), columns = hour (0-23)."""
    matrix = df.pivot_table(
        index=df["time"].dt.day_of_week,
        columns=df["time"].dt.hour,
        values="aqi",
        aggfunc="mean",
        dropna=False,
    )
    return matrix.reindex(index=range(7), columns=range(24)).rename_axis(
        index="weekday", columns="hour"
    )


def monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly means: ``year``, ``month``, ``pm25``, ``aqi``."""
    grouped = df.groupby([df["time"].dt.year, df["time"].dt.month])[METRICS].mean()
    return grouped.rename_axis(["year", "month"]).reset_index()


def summary(df: pd.DataFrame) -> dict:
    """Key numbers for the "now" card.

    Keys: ``last_time``, ``last_pm25``, ``last_aqi`` (latest row with a valid AQI),
    ``mean_pm25_30d``, ``mean_aqi_30d``, ``worst_day``, ``worst_day_aqi``.
    The 30-day window ends at the latest measurement (data may be older than today);
    the worst day is also searched inside this window. Empty data gives ``None`` values.
    """
    result = dict.fromkeys(
        ["last_time", "last_pm25", "last_aqi", "mean_pm25_30d", "mean_aqi_30d",
         "worst_day", "worst_day_aqi"]
    )
    valid = df.dropna(subset=["aqi"])
    if valid.empty:
        return result
    last = valid.iloc[-1]
    window = last_days(valid, SUMMARY_WINDOW_DAYS)
    days = daily(window).dropna(subset=["aqi"])
    result.update(
        last_time=last["time"],
        last_pm25=last["pm25"],
        last_aqi=last["aqi"],
        mean_pm25_30d=window["pm25"].mean(),
        mean_aqi_30d=window["aqi"].mean(),
    )
    if not days.empty:
        worst = days.loc[days["aqi"].idxmax()]
        result.update(worst_day=worst["date"], worst_day_aqi=worst["aqi"])
    return result


def weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly means (weeks start on Monday): ``week``, ``pm25``, ``aqi``."""
    grouped = df.set_index("time")[METRICS].resample("W-MON", label="left", closed="left").mean()
    return grouped.dropna(how="all").rename_axis("week").reset_index()


def yearly(df: pd.DataFrame) -> pd.DataFrame:
    """Yearly means: ``year``, ``pm25``, ``aqi``, ``days`` (days with data)."""
    days = daily(df).dropna(subset=["aqi"])
    grouped = days.groupby(days["date"].dt.year).agg(
        pm25=("pm25", "mean"), aqi=("aqi", "mean"), days=("aqi", "size")
    )
    return grouped.rename_axis("year").reset_index()


def month_year_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Mean AQI: rows = year, columns = month (1-12)."""
    table = monthly(df)
    matrix = table.pivot(index="year", columns="month", values="aqi")
    return matrix.reindex(columns=range(1, 13))


def city_stats(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per city over its last year of data.

    Columns: ``city``, ``mean_aqi``, ``median_aqi``, ``p95_aqi``,
    ``unhealthy_share`` (% of hours with AQI > 100), ``best_hour``, ``worst_hour``.
    """
    rows = []
    for city, df in dfs.items():
        recent = last_days(df, STATS_WINDOW_DAYS).dropna(subset=["aqi"])
        if recent.empty:
            continue
        profile = hourly_profile(recent).dropna(subset=["aqi"])
        rows.append({
            "city": city,
            "mean_aqi": recent["aqi"].mean(),
            "median_aqi": recent["aqi"].median(),
            "p95_aqi": recent["aqi"].quantile(0.95),
            "unhealthy_share": (recent["aqi"] > UNHEALTHY_AQI).mean() * 100,
            "best_hour": int(profile.loc[profile["aqi"].idxmin(), "hour"]),
            "worst_hour": int(profile.loc[profile["aqi"].idxmax(), "hour"]),
        })
    columns = ["city", "mean_aqi", "median_aqi", "p95_aqi", "unhealthy_share", "best_hour", "worst_hour"]
    return pd.DataFrame(rows, columns=columns)


def city_hour_matrix(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean AQI over the last year: rows = city, columns = hour (0-23)."""
    rows = {}
    for city, df in dfs.items():
        recent = last_days(df, STATS_WINDOW_DAYS)
        if recent["aqi"].notna().any():
            rows[city] = hourly_profile(recent).set_index("hour")["aqi"]
    return pd.DataFrame(rows).T.reindex(columns=range(24)) if rows else pd.DataFrame()


def daily_by_city(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily AQI of every city over its last year, long format: ``city, date, aqi``."""
    frames = [
        daily(last_days(df, STATS_WINDOW_DAYS)).dropna(subset=["aqi"]).assign(city=city)
        for city, df in dfs.items() if not df.empty
    ]
    if not frames:
        return pd.DataFrame(columns=["city", "date", "aqi"])
    return pd.concat(frames, ignore_index=True)[["city", "date", "aqi"]]
