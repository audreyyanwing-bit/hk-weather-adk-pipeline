"""Central configuration. Values come from environment variables / the .env file.

No secrets are hard-coded here: the Gemini key and bucket name live in .env
(which is git-ignored), and Google Cloud credentials come from
`gcloud auth application-default login`.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# --- Model -------------------------------------------------------------------
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# --- Data source: Hong Kong Observatory Open Data API (no key required) -------
HKO_CLIMATE_URL = "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php"
HKO_WEATHER_URL = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php"

# Station codes that publish daily mean temperature (CLMTEMP), per the HKO
# Open Data API documentation, with friendly names for the commonly used ones.
STATION_CODES = [
    "CCH", "CWB", "HKA", "HKO", "HKP", "HKS", "HPV", "JKB", "KLT", "KP", "KSC",
    "KTG", "LFS", "NGP", "PEN", "PLC", "SE1", "SEK", "SHA", "SKG", "SKW", "SSH",
    "SSP", "STY", "TC", "TKL", "TMS", "TPO", "TU1", "TW", "TWN", "TY1", "TYW",
    "VP1", "WGL", "WLP", "WTS", "YCT", "YLP",
]
STATION_NAMES = {
    "HKO": "Hong Kong Observatory headquarters",
    "KP": "King's Park",
    "HKA": "Hong Kong International Airport",
    "CCH": "Cheung Chau",
    "SHA": "Sha Tin",
    "TKL": "Ta Kwu Ling",
    "LFS": "Lau Fau Shan",
    "TMS": "Tai Mo Shan",
    "WGL": "Waglan Island",
    "SKG": "Sai Kung",
    "TPO": "Tai Po",
    "STY": "Stanley",
}


def station_label(code: str) -> str:
    name = STATION_NAMES.get(code)
    return f"{code} - {name}" if name else code


DEFAULT_STATION = "HKO"

# --- Storage -----------------------------------------------------------------
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "output"))
# Google Cloud project that owns the bucket (needed by the storage client)
GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip() or None
GCS_BUCKET = os.getenv("GCS_BUCKET", "").strip().removeprefix("gs://").strip("/")
GCS_PREFIX = os.getenv("GCS_PREFIX", "hk-weather").strip("/")


def new_run_id() -> str:
    """A sortable, unique id for one pipeline run, e.g. 20261113T201500Z."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
