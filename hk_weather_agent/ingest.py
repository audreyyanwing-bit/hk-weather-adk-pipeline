"""Step 1 - Ingestion: pull raw data from the Hong Kong Observatory Open Data API."""

import json
import time
from pathlib import Path

import requests
from google.adk.tools import ToolContext

from . import config


def _get_json(url: str, params: dict, attempts: int = 3) -> dict:
    """GET a JSON endpoint with a timeout and simple exponential-backoff retries."""
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as err:
            last_error = err
            if attempt < attempts:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"HKO API request failed after {attempts} attempts: {last_error}")


def _save_json(data: dict, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(path)


def fetch_hko_data(station: str, tool_context: ToolContext) -> dict:
    """Downloads raw Hong Kong Observatory data and saves it to the run folder.

    Fetches two datasets:
      1. Daily mean temperature history (CLMTEMP) for one weather station.
      2. The current 9-day weather forecast for Hong Kong.

    Args:
        station: HKO station code, e.g. "HKO" (Observatory headquarters,
            records since 1884), "KP" (King's Park) or "HKA" (Airport).
            Use "HKO" if the user did not ask for a specific station.

    Returns:
        A short summary: status, run id, record counts and saved file paths.
    """
    station = (station or config.DEFAULT_STATION).strip().upper()
    if station not in config.STATION_CODES:
        return {
            "status": "error",
            "message": f"Unknown station code '{station}'. "
            f"Valid codes include: {', '.join(config.STATION_CODES)}.",
        }

    run_id = config.new_run_id()
    run_dir = config.OUTPUT_DIR / "runs" / run_id
    raw_dir = run_dir / "raw"

    try:
        climate = _get_json(
            config.HKO_CLIMATE_URL,
            {"dataType": "CLMTEMP", "rformat": "json", "station": station},
        )
        forecast = _get_json(config.HKO_WEATHER_URL, {"dataType": "fnd", "lang": "en"})
    except RuntimeError as err:
        return {"status": "error", "message": str(err)}

    if not climate.get("data"):
        return {"status": "error", "message": f"No temperature records returned for {station}."}

    raw_files = {
        "climate": _save_json(climate, raw_dir / f"clmtemp_{station}.json"),
        "forecast": _save_json(forecast, raw_dir / "forecast_9day.json"),
    }

    # Share locations with the downstream agents through session state.
    tool_context.state["run_id"] = run_id
    tool_context.state["run_dir"] = str(run_dir)
    tool_context.state["station"] = station
    tool_context.state["raw_files"] = raw_files

    first, last = climate["data"][0], climate["data"][-1]
    return {
        "status": "success",
        "run_id": run_id,
        "station": config.station_label(station),
        "temperature_records": len(climate["data"]),
        "date_range": f"{first[0]}-{first[1]}-{first[2]} to {last[0]}-{last[1]}-{last[2]}",
        "forecast_days": len(forecast.get("weatherForecast", [])),
        "raw_files": raw_files,
    }
