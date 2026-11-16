"""City config: where the CSV files live and how each city is called on the site.

A city is any ``data/<city>.csv`` file; ``CITY_LABELS`` maps the file name to its Ukrainian name.
"""
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
CITY_LABELS = {
    "dnipro": "Дніпро", "kharkiv": "Харків", "kyiv": "Київ",
    "lutsk": "Луцьк", "lviv": "Львів", "odessa": "Одеса",
}


def city_label(city: str) -> str:
    """Ukrainian display name of a city (falls back to the capitalised file name)."""
    return CITY_LABELS.get(city, city.title())
