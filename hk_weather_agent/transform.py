"""Step 2 - Processing: clean, filter, group and aggregate the raw HKO data.

The pure functions (parse_*, clean_*, build_*) hold the logic and are unit
tested; `transform_weather_data` is the thin ADK tool wrapper around them.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from google.adk.tools import ToolContext

from . import config

RAIN_LIKELY = {"High", "Medium High"}


# ---------------------------------------------------------------- parsing ----
def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan  # HKO uses "***" for unavailable values


def parse_daily_temperature(raw: dict) -> pd.DataFrame:
    """CLMTEMP JSON -> one row per day: year, month, day, mean_temp, complete.

    HKO row layout: [year, month, day, value, completeness], where
    completeness "C" means the day's record is complete.
    """
    rows = []
    for row in raw.get("data", []):
        if len(row) < 4:
            continue
        try:
            year, month, day = int(row[0]), int(row[1]), int(row[2])
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "year": year,
                "month": month,
                "day": day,
                "mean_temp": _to_float(row[3]),
                "complete": (row[4] if len(row) > 4 else "") == "C",
            }
        )
    return pd.DataFrame(rows, columns=["year", "month", "day", "mean_temp", "complete"])


def clean_daily_temperature(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Filter out missing/incomplete/invalid days and duplicates. Returns (clean, quality report)."""
    total = len(df)
    missing = int(df["mean_temp"].isna().sum())
    incomplete = int((~df["complete"] & df["mean_temp"].notna()).sum())

    clean = df[df["complete"] & df["mean_temp"].notna()].copy()
    # Plausibility check: Hong Kong daily means sit well inside 0-40 C.
    implausible = int((~clean["mean_temp"].between(0, 40)).sum())
    clean = clean[clean["mean_temp"].between(0, 40)]

    clean["date"] = pd.to_datetime(clean[["year", "month", "day"]], errors="coerce")
    bad_dates = int(clean["date"].isna().sum())
    clean = clean.dropna(subset=["date"])
    duplicates = int(clean.duplicated(subset=["date"]).sum())
    clean = clean.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    quality = {
        "rows_received": total,
        "rows_kept": len(clean),
        "dropped_missing_value": missing,
        "dropped_incomplete_flag": incomplete,
        "dropped_implausible_value": implausible,
        "dropped_bad_date": bad_dates,
        "dropped_duplicates": duplicates,
    }
    return clean[["date", "year", "month", "day", "mean_temp"]], quality


def parse_forecast(raw: dict) -> pd.DataFrame:
    """9-day forecast JSON -> one row per forecast day."""

    def val(x):
        return x.get("value") if isinstance(x, dict) else x

    rows = []
    for item in raw.get("weatherForecast", []):
        rows.append(
            {
                "date": pd.to_datetime(str(item.get("forecastDate")), format="%Y%m%d", errors="coerce"),
                "weekday": item.get("week", ""),
                "min_temp": _to_float(val(item.get("forecastMintemp"))),
                "max_temp": _to_float(val(item.get("forecastMaxtemp"))),
                "min_rh": _to_float(val(item.get("forecastMinrh"))),
                "max_rh": _to_float(val(item.get("forecastMaxrh"))),
                "rain_probability": item.get("PSR", ""),
                "weather": item.get("forecastWeather", ""),
            }
        )
    return pd.DataFrame(rows).dropna(subset=["date"]) if rows else pd.DataFrame(rows)


# ------------------------------------------------------------ aggregation ----
def build_annual_summary(
    daily: pd.DataFrame, baseline: tuple[int, int], min_days: int
) -> pd.DataFrame:
    """Group by year -> annual mean temperature, anomaly vs baseline, 10-year rolling mean.

    Years with fewer than `min_days` complete days (e.g. the current partial
    year, or 1940-1946 when records were interrupted) are excluded so they
    do not distort the trend.
    """
    annual = (
        daily.groupby("year")["mean_temp"]
        .agg(mean_temp="mean", days="count", max_daily_mean="max", min_daily_mean="min")
        .reset_index()
    )
    annual = annual[annual["days"] >= min_days].copy()
    base = annual[annual["year"].between(*baseline)]["mean_temp"].mean()
    annual["anomaly_vs_baseline"] = annual["mean_temp"] - base
    annual["rolling_10yr_mean"] = annual["mean_temp"].rolling(10, min_periods=8).mean()
    return annual.round(2).reset_index(drop=True)


def build_monthly_comparison(daily: pd.DataFrame, baseline: tuple[int, int]) -> tuple[pd.DataFrame, int]:
    """Monthly normals over the baseline vs. the most recent year's monthly means."""
    base = daily[daily["year"].between(*baseline)]
    normals = base.groupby("month")["mean_temp"].mean().rename("baseline_normal")

    latest_year = int(daily["year"].max())
    latest = daily[daily["year"] == latest_year].groupby("month")["mean_temp"].agg(["mean", "count"])
    latest = latest[latest["count"] >= 15]["mean"].rename("latest_year_mean")  # skip thin months

    monthly = pd.concat([normals, latest], axis=1).reindex(range(1, 13))
    monthly["anomaly"] = monthly["latest_year_mean"] - monthly["baseline_normal"]
    monthly.index.name = "month"
    return monthly.round(2).reset_index(), latest_year


def warming_trend(annual: pd.DataFrame, since: int | None = None) -> float:
    """Linear-regression slope of annual mean temperature, in C per decade."""
    data = annual if since is None else annual[annual["year"] >= since]
    if len(data) < 5:
        return float("nan")
    slope = np.polyfit(data["year"], data["mean_temp"], 1)[0]
    return round(float(slope) * 10, 3)


def enrich_forecast(forecast: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """Add daily range, a rain flag, and how warm each day is vs. its monthly normal."""
    if forecast.empty:
        return forecast
    normals = monthly.set_index("month")["baseline_normal"]
    out = forecast.copy()
    out["temp_range"] = out["max_temp"] - out["min_temp"]
    out["forecast_mean"] = (out["max_temp"] + out["min_temp"]) / 2
    out["vs_monthly_normal"] = out["forecast_mean"] - out["date"].dt.month.map(normals)
    out["rain_likely"] = out["rain_probability"].isin(RAIN_LIKELY)
    return out.round({c: 2 for c in out.select_dtypes("number").columns})


# ----------------------------------------------------------------- ADK tool ---
def transform_weather_data(
    tool_context: ToolContext,
    baseline_start: int = 1991,
    baseline_end: int = 2020,
    min_days_per_year: int = 330,
) -> dict:
    """Cleans and aggregates the raw data saved by the ingestion step.

    Filters out incomplete or invalid days, groups daily temperatures by year
    and by month, compares them with a climate baseline period, computes the
    long-term warming trend, and enriches the 9-day forecast.

    Args:
        baseline_start: First year of the climate "normal" period (default 1991).
        baseline_end: Last year of the climate "normal" period (default 2020).
        min_days_per_year: Minimum complete days for a year to count (default 330).

    Returns:
        Key metrics for the visualization step, plus processed file paths.
    """
    raw_files = tool_context.state.get("raw_files")
    run_dir = tool_context.state.get("run_dir")
    if not raw_files or not run_dir:
        return {"status": "error", "message": "No raw data found - the ingestion step must succeed first."}
    if baseline_start >= baseline_end:
        return {"status": "error", "message": "baseline_start must be earlier than baseline_end."}

    climate_raw = json.loads(Path(raw_files["climate"]).read_text(encoding="utf-8"))
    forecast_raw = json.loads(Path(raw_files["forecast"]).read_text(encoding="utf-8"))
    baseline = (baseline_start, baseline_end)

    daily, quality = clean_daily_temperature(parse_daily_temperature(climate_raw))
    if daily.empty:
        return {"status": "error", "message": "No usable temperature records after cleaning."}

    annual = build_annual_summary(daily, baseline, min_days_per_year)
    monthly, latest_year = build_monthly_comparison(daily, baseline)
    forecast = enrich_forecast(parse_forecast(forecast_raw), monthly)
    if annual.empty:
        return {"status": "error", "message": "No year has enough complete days to summarise."}

    out_dir = Path(run_dir) / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "daily_clean": out_dir / "daily_clean.csv",
        "annual_summary": out_dir / "annual_summary.csv",
        "monthly_comparison": out_dir / "monthly_comparison.csv",
        "forecast_9day": out_dir / "forecast_9day.csv",
        "metrics": out_dir / "metrics.json",
    }
    daily.to_csv(files["daily_clean"], index=False, date_format="%Y-%m-%d")
    annual.to_csv(files["annual_summary"], index=False)
    monthly.to_csv(files["monthly_comparison"], index=False)
    forecast.to_csv(files["forecast_9day"], index=False, date_format="%Y-%m-%d")

    warmest = annual.nlargest(5, "mean_temp")[["year", "mean_temp"]]
    latest_months = monthly.dropna(subset=["latest_year_mean"])
    station = tool_context.state.get("station", config.DEFAULT_STATION)
    metrics = {
        "station": config.station_label(station),
        "record_years": f"{int(daily['year'].min())}-{int(daily['year'].max())}",
        "baseline_period": f"{baseline_start}-{baseline_end}",
        "baseline_annual_mean_c": round(
            float(annual[annual["year"].between(*baseline)]["mean_temp"].mean()), 2
        ),
        "years_summarised": int(len(annual)),
        "warming_trend_c_per_decade_full_record": warming_trend(annual),
        "warming_trend_c_per_decade_since_1991": warming_trend(annual, since=1991),
        "warmest_5_years": [
            {"year": int(r.year), "mean_temp_c": float(r.mean_temp)} for r in warmest.itertuples()
        ],
        "latest_full_year": int(annual["year"].max()),
        "latest_full_year_anomaly_c": float(annual.iloc[-1]["anomaly_vs_baseline"]),
        "current_year": latest_year,
        "current_year_months_above_normal": int((latest_months["anomaly"] > 0).sum()),
        "current_year_months_with_data": int(len(latest_months)),
        "current_year_avg_monthly_anomaly_c": round(float(latest_months["anomaly"].mean()), 2)
        if len(latest_months)
        else None,
        "forecast_days": int(len(forecast)),
        "forecast_rain_likely_days": int(forecast["rain_likely"].sum()) if len(forecast) else 0,
        "forecast_avg_vs_normal_c": round(float(forecast["vs_monthly_normal"].mean()), 2)
        if len(forecast)
        else None,
        "data_quality": quality,
    }
    files["metrics"].write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    tool_context.state["processed_files"] = {k: str(v) for k, v in files.items()}
    tool_context.state["metrics_summary"] = json.dumps(metrics)
    return {"status": "success", "metrics": metrics, "processed_files": tool_context.state["processed_files"]}
