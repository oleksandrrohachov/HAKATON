import pandas as pd
import pytest

import loader


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    csv = (
        "phenomenon,value,logged_at\n"
        'pm25,10.0,"2024-01-01 00:00:00"\n'
        'pm25,12.0,"2024-01-01 00:00:00"\n'  # duplicate hour -> averaged
        'pm25,-5.0,"2024-01-01 01:00:00"\n'  # negative -> NaN
        'pm25,2000,"2024-01-01 02:00:00"\n'  # outlier -> NaN
        'pm25,abc,"2024-01-01 03:00:00"\n'  # non-numeric -> NaN
        'pm25,30.0,"2024-01-01 04:00:00"\n'
        'temperature,5.0,"2024-01-01 00:00:00"\n'  # ignored
    )
    (tmp_path / "testcity.csv").write_text(csv)
    monkeypatch.setattr(loader, "DATA_DIR", tmp_path)
    return tmp_path


def test_load_city_cleans_and_computes_aqi(data_dir):
    df = loader.load_city("TestCity")  # case-insensitive
    assert list(df.columns) == ["time", "pm25", "aqi"]
    assert df["time"].is_monotonic_increasing and not df["time"].duplicated().any()
    assert df.loc[0, "pm25"] == 11.0
    assert df["pm25"].isna().sum() == 2  # negative and outlier masked, unparsable row dropped
    assert len(df) == 4
    assert df["aqi"].iloc[0] == pytest.approx(46, abs=1)  # EPA: pm25 11 -> ~46


def test_unknown_city_lists_available(data_dir):
    with pytest.raises(FileNotFoundError, match="testcity"):
        loader.load_city("paris")


def test_pm25_to_aqi_breakpoints():
    result = loader.pm25_to_aqi(pd.Series([0.0, 12.0, 35.4, float("nan")]))
    assert result.iloc[:3].tolist() == [0, 50, 100] and pd.isna(result.iloc[3])
