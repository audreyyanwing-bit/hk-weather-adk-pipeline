"""Unit tests for the processing logic."""

import pandas as pd

from hk_weather_agent import transform as t


def test_parse_handles_missing_and_flags(clmtemp):
    df = t.parse_daily_temperature(clmtemp)
    assert len(df) == len(clmtemp["data"])
    assert df["mean_temp"].isna().sum() > 0          # "***" -> NaN
    assert (~df["complete"]).sum() > 0                # "#" / "" -> incomplete


def test_clean_drops_bad_rows_and_reports(clmtemp):
    clean, q = t.clean_daily_temperature(t.parse_daily_temperature(clmtemp))
    assert q["rows_received"] == len(clmtemp["data"])
    assert q["rows_kept"] == len(clean)
    assert q["dropped_missing_value"] > 0 and q["dropped_incomplete_flag"] > 0
    assert clean["mean_temp"].between(0, 40).all()
    assert clean["date"].is_monotonic_increasing and not clean["date"].duplicated().any()


def test_clean_removes_implausible_and_duplicate_rows():
    raw = {"data": [["2020", "1", "1", "15.0", "C"], ["2020", "1", "1", "15.0", "C"],
                    ["2020", "1", "2", "99.9", "C"], ["2020", "2", "30", "15.0", "C"]]}
    clean, q = t.clean_daily_temperature(t.parse_daily_temperature(raw))
    assert len(clean) == 1
    assert q["dropped_implausible_value"] == 1 and q["dropped_duplicates"] == 1 and q["dropped_bad_date"] == 1


def test_annual_summary_excludes_partial_years(clmtemp):
    daily, _ = t.clean_daily_temperature(t.parse_daily_temperature(clmtemp))
    annual = t.build_annual_summary(daily, (1991, 2020), min_days=330)
    assert 2026 not in annual["year"].values          # partial current year excluded
    assert abs(annual[annual["year"].between(1991, 2020)]["anomaly_vs_baseline"].mean()) < 0.01
    assert annual["rolling_10yr_mean"].notna().any()


def test_warming_trend_recovers_synthetic_slope(clmtemp):
    daily, _ = t.clean_daily_temperature(t.parse_daily_temperature(clmtemp))
    annual = t.build_annual_summary(daily, (1991, 2020), 330)
    assert abs(t.warming_trend(annual) - 0.2) < 0.03  # fixture warms 0.2 C/decade


def test_monthly_comparison_shape(clmtemp):
    daily, _ = t.clean_daily_temperature(t.parse_daily_temperature(clmtemp))
    monthly, latest_year = t.build_monthly_comparison(daily, (1991, 2020))
    assert latest_year == 2026 and len(monthly) == 12
    assert pd.isna(monthly.loc[monthly["month"] == 12, "latest_year_mean"]).all()  # no Dec data yet
    assert (monthly.dropna()["anomaly"] > 0).all()   # warming fixture -> above normal


def test_forecast_enrichment(clmtemp, forecast_raw):
    daily, _ = t.clean_daily_temperature(t.parse_daily_temperature(clmtemp))
    monthly, _ = t.build_monthly_comparison(daily, (1991, 2020))
    fc = t.enrich_forecast(t.parse_forecast(forecast_raw), monthly)
    assert len(fc) == 9
    assert (fc["temp_range"] > 0).all()
    assert fc["rain_likely"].sum() == 2               # "Medium High" + "High"
    assert fc["vs_monthly_normal"].notna().all()
