# Контракт функцій

`df` — погодинна таблиця з `loader.load_city`: колонки `time` (datetime), `pm25`, `aqi` (пропуски = NaN).
`dfs` — `dict[str, DataFrame]`, місто → `df`.

## cities.py — список міст

| Назва | Що це |
|---|---|
| `DATA_DIR` | папка `data/` з CSV (`data/<city>.csv`) |
| `CITY_LABELS` | `dict[str, str]`, ім'я файлу → українська назва |
| `city_label(city)` | `str` → `str`, «kyiv» → «Київ» |

## loader.py — дані

| Функція | Вхід | Вихід |
|---|---|---|
| `list_cities()` | — | `list[str]`, імена файлів `data/*.csv` без розширення |
| `city_label(city)` | `str` | те саме, що `cities.city_label` (для зручності) |
| `load_city(city)` | `city: str` (регістр неважливий) | `df`, відсортований, без дублів. `FileNotFoundError` зі списком міст, `ValueError` для битого файлу |
| `pm25_to_aqi(pm25)` | `pd.Series` | `pd.Series` AQI за таблицею US EPA (`EPA_PM25_BREAKPOINTS`) |

## analysis.py — аналітика (що було)

| Функція | Вхід | Вихід |
|---|---|---|
| `last_days(df, days)` | `df`, `int` | рядки останніх `days` днів (до останнього вимірювання) |
| `daily(df)` | `df` | `DataFrame[date, pm25, aqi]` |
| `weekly(df)` | `df` | `DataFrame[week, pm25, aqi]`, тижні з понеділка |
| `monthly(df)` | `df` | `DataFrame[year, month, pm25, aqi]` |
| `yearly(df)` | `df` | `DataFrame[year, pm25, aqi, days]` |
| `hourly_profile(df)` | `df` | `DataFrame[hour 0–23, pm25, aqi]` |
| `weekday_profile(df)` | `df` | `DataFrame[weekday 0=Пн…6=Нд, pm25, aqi]` |
| `hour_weekday_matrix(df)` | `df` | 7×24: день тижня × година, середній AQI |
| `month_year_matrix(df)` | `df` | рік × місяць 1–12, середній AQI |
| `summary(df)` | `df` | `dict`: `last_time`, `last_pm25`, `last_aqi`, `mean_pm25_30d`, `mean_aqi_30d`, `worst_day`, `worst_day_aqi` |
| `city_stats(dfs)` | `dfs` | `DataFrame[city, mean_aqi, median_aqi, p95_aqi, unhealthy_share, best_hour, worst_hour]` за останній рік |
| `city_hour_matrix(dfs)` | `dfs` | місто × година 0–23, середній AQI за рік |
| `daily_by_city(dfs)` | `dfs` | `DataFrame[city, date, aqi]` за останній рік |

## insights.py — тексти

| Функція | Вхід | Вихід |
|---|---|---|
| `aqi_band(aqi)` | `float` | `int` 0–5 або `None` для NaN |
| `aqi_category(aqi)` | `float` | «Добре» … «Небезпечно» або «Немає даних» |
| `advice(aqi)` | `float` | коротка порада |
| `best_hour(hourly)` | `hourly_profile(df)` | `"06:00"` |
| `compare_to_last_year(df, month)` | `df`, `int` 1–12 | «на 12% чистіше, ніж торік» |
| `category_share_by_year(df)` | `df` | рік × категорія 0–5, % днів |
| `key_findings(df)` | `df` | `list[str]`, прості висновки про місто |

## forecast.py — короткостроковий прогноз (дні)

| Функція | Вхід | Вихід |
|---|---|---|
| `daily_series(df, max_days=365)` | `df` | `pd.Series` добового AQI без пропусків (останній безперервний відрізок) |
| `forecast_aqi(df, days=7)` | `df`, `int` | `DataFrame[date, yhat, lower, upper]`, `0 ≤ lower ≤ yhat ≤ upper`. `ValueError`, якщо даних < 10 днів |
| `rolling_backtest(df, days=7, folds=8)` | `df` | `DataFrame[date, actual, naive, seasonal_naive, holt_winters]` |
| `backtest_mae(df, days=7)` | `df` | `float`, MAE на останніх `days` днях |
| `model_errors(backtest)` | таблиця з `actual` і колонками моделей | `DataFrame[model, mae]`, найкраща першою |
| `decompose_daily(df, period=365)` | `df` | `DataFrame[date, observed, trend, seasonal, resid]`; `ValueError`, якщо < `2 × period` днів |

## seasonal.py — сезонна модель і перевірка 80/20 (тижні)

| Функція | Вхід | Вихід |
|---|---|---|
| `weekly_series(df)` | `df` | `pd.Series` тижневого AQI (лише повні тижні) |
| `seasonal_split(df, train_share=0.8)` | `df` | `(train[week, aqi], test[week, actual, train_mean, last_year, seasonal_hw])`; `ValueError`, якщо навчальних тижнів < 104 |
| `seasonal_forecast(df, weeks=26)` | `df` | `DataFrame[date, yhat, lower, upper]` |

## charts/ — графіки (усі повертають `plotly.graph_objects.Figure`, на порожніх даних — порожній Figure з підписом)

| Модуль | Функції |
|---|---|
| `charts/base.py` | `color_for_aqi(aqi) -> str`, `aqi_colors`, `empty_figure`, `gradient_color`, `aqi_colorscale`, `aqi_heatmap` |
| `charts/analytics.py` | `weekly_trend(weekly)`, `hourly_profile_chart(hourly)`, `weekday_profile_chart(weekday)`, `heatmap_hour_weekday(matrix)`, `heatmap_month_year(matrix)`, `category_share_chart(shares)`, `decomposition_chart(parts)` |
| `charts/prediction.py` | `line_with_forecast(history, forecast, title)`, `train_test_chart(train, test, model)`, `predicted_vs_actual(backtest, model)`, `model_comparison(errors, title)` |
| `charts/cities.py` | `compare_cities(dfs)`, `city_boxplot(daily_by_city)`, `heatmap_city_hour(matrix)` |

`import charts` дає доступ до всіх функцій (див. `charts/__init__.py`).

## glossary.py — пояснення

| Назва | Що це |
|---|---|
| `TERMS` | `dict[str, str]`, термін → пояснення |
| `CHART_HELP` | `dict[str, str]`, ключ графіка → підказка під графіком |
| `ANALYTICS_INTRO`, `FORECAST_INTRO` | вступ до вкладок |

## ui/ — сайт (Streamlit)

| Модуль | Функція |
|---|---|
| `ui/data.py` | кешовані `city_data`, `all_cities`, `short_forecast`, `short_backtest`, `decomposition`, `seasonal_check`, `seasonal_outlook`, `daily` |
| `ui/widgets.py` | `chart(fig, help_key, container)` — графік + підказка |
| `ui/now_card.py` | `render(df)` — картка «Зараз» |
| `ui/analytics_tab.py` | `render(city, df)` — вкладка «Аналітика» |
| `ui/forecast_tab.py` | `render(city, days)` — вкладка «Прогноз» |
| `ui/cities_tab.py` | `render(cities)` — вкладка «Порівняння міст» |
| `ui/glossary_tab.py` | `render()` — вкладка «Словник» |
| `app.py` | `main()` — бічна панель, вкладки, футер |

## report.py — звет

| Функція | Вхід | Вихід |
|---|---|---|
| `analytics_figures(df)` / `forecast_figures(df)` / `cities_figures(dfs)` | дані | `dict[str, Figure]`, ім'я файлу → графік |
| `build_figures(city, dfs)` | місто, `dfs` | усі 16 графіків по порядку |
| `save_figures(figures, plots_dir)` | графіки, папка | PNG-файли |
| `short_term_errors(dfs)` / `seasonal_errors(dfs)` | `dfs` | MAE: місто × модель |
| `build_summary(city, dfs)` | місто, `dfs` | `str`: розділ А «Аналітика» і розділ Б «Прогноз» |
| `main(city="kyiv")` | місто | `outputs/plots/*.png`, `outputs/reports/summary.txt`, `city_stats.csv` |
