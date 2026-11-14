"""Shared test fixtures: small synthetic HKO responses (no network or API key needed)."""

import math
from datetime import date, timedelta

import pytest

from hk_weather_agent import config


def make_clmtemp(start=date(1980, 1, 1), end=date(2026, 11, 12)) -> dict:
    """Synthetic CLMTEMP response in the real layout: [year, month, day, value, completeness]."""
    rows, d, i = [], start, 0
    while d <= end:
        seasonal = 23.5 - 6.5 * math.cos(2 * math.pi * (d.timetuple().tm_yday - 15) / 365)
        warming = 0.02 * (d.year - start.year)  # +0.2 C per decade
        value, flag = f"{seasonal + warming:.1f}", "C"
        if i % 97 == 0:
            value, flag = "***", ""  # missing value
        elif i % 131 == 0:
            flag = "#"  # incomplete record
        rows.append([str(d.year), str(d.month), str(d.day), value, flag])
        d += timedelta(days=1)
        i += 1
    return {"type": ["CLMTEMP"], "fields": ["Year", "Month", "Day", "Value", "Completeness"],
            "data": rows, "legend": []}


def make_forecast(start=date(2026, 11, 14), days=9) -> dict:
    psr = ["Low", "Low", "Medium Low", "Medium", "Medium High", "High", "Medium", "Low", "Low"]
    items = []
    for k in range(days):
        d = start + timedelta(days=k)
        items.append({
            "forecastDate": d.strftime("%Y%m%d"), "week": d.strftime("%A"),
            "forecastWind": "East force 4.", "forecastWeather": "Mainly cloudy.",
            "forecastMaxtemp": {"value": 25 + k % 3, "unit": "C"},
            "forecastMintemp": {"value": 20 + k % 2, "unit": "C"},
            "forecastMaxrh": {"value": 90, "unit": "percent"},
            "forecastMinrh": {"value": 65, "unit": "percent"},
            "ForecastIcon": 60, "PSR": psr[k % len(psr)],
        })
    return {"generalSituation": "Test.", "weatherForecast": items, "updateTime": "2026-11-13T16:30:00+08:00"}


@pytest.fixture
def clmtemp():
    return make_clmtemp()


@pytest.fixture
def forecast_raw():
    return make_forecast()


@pytest.fixture
def isolated_output(tmp_path, monkeypatch):
    """Write outputs to a temp folder and disable GCS uploads."""
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(config, "GCS_BUCKET", "")
    return tmp_path
