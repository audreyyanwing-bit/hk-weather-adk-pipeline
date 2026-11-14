"""Agent definitions for the Hong Kong weather pipeline (Google ADK).

Architecture
------------
root_agent (SequentialAgent "hk_weather_pipeline") runs four LLM agents in order.
Each agent owns one tool and passes results to the next through session state:

    ingestion_agent      -> fetch_hko_data             (HKO Open Data API -> raw JSON)
    processing_agent     -> transform_weather_data     (clean, group, aggregate -> CSV + metrics)
    visualization_agent  -> build_dashboard            (writes the summary + HTML dashboard)
    storage_agent        -> upload_to_cloud_storage    (raw + processed + dashboard -> GCS)

The agents decide the tool arguments from the user's request (e.g. which
station, which baseline period), the visualization agent writes the
dashboard's headline and insights from the computed metrics, and each agent
checks its tool's result and reports problems instead of continuing silently.
"""

import warnings

from google.adk.agents import LlmAgent, SequentialAgent
from google.genai import types

from . import config
from .ingest import fetch_hko_data
from .storage import upload_to_cloud_storage
from .transform import transform_weather_data
from .visualize import build_dashboard

# Free-tier Gemini keys are rate-limited; retry 429/5xx errors with backoff.
GEN_CONFIG = types.GenerateContentConfig(
    temperature=0.2,
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(
            attempts=6, initial_delay=5.0, max_delay=60.0, exp_base=2.0,
            http_status_codes=[429, 500, 502, 503, 504],
        )
    ),
)

STATION_LIST = ", ".join(f"{code} ({name})" for code, name in config.STATION_NAMES.items())

ingestion_agent = LlmAgent(
    name="ingestion_agent",
    model=config.MODEL,
    description="Downloads raw Hong Kong Observatory data.",
    instruction=f"""You are the data ingestion step of a weather data pipeline.
Read the user's request and decide which Hong Kong Observatory station to use.
Common station codes: {STATION_LIST}.
If the user did not name a station, use "HKO".
Call the `fetch_hko_data` tool exactly once with that station code.
If the tool returns status "error", explain the problem in one sentence and stop.
Otherwise reply with one sentence: the station, number of records, date range and run id.""",
    tools=[fetch_hko_data],
    output_key="ingestion_report",
    generate_content_config=GEN_CONFIG,
)

processing_agent = LlmAgent(
    name="processing_agent",
    model=config.MODEL,
    description="Cleans and aggregates the raw data into yearly, monthly and forecast tables.",
    instruction="""You are the processing step of a weather data pipeline.
Ingestion result: {ingestion_report?}
If ingestion failed, reply "Skipped: no raw data." and do not call any tool.
Otherwise call `transform_weather_data` exactly once.
Use the default 1991-2020 climate baseline unless the user asked for a different
baseline period, in which case pass baseline_start and baseline_end.
Then reply with 2-3 sentences covering the data-quality results (rows kept vs dropped)
and the headline numbers (warming trend, latest full year anomaly).""",
    tools=[transform_weather_data],
    output_key="processing_report",
    generate_content_config=GEN_CONFIG,
)

visualization_agent = LlmAgent(
    name="visualization_agent",
    model=config.MODEL,
    description="Writes the summary and builds the HTML dashboard.",
    instruction="""You are the visualization and reporting step of a weather data pipeline.
Computed metrics (JSON): {metrics_summary?}
If there are no metrics, reply "Skipped: no processed data." and do not call any tool.
Otherwise call `build_dashboard` exactly once with:
  - headline: one plain-English headline of at most 15 words about the main finding.
  - insights: 3 or 4 short bullet points for a non-technical reader. Use ONLY numbers
    that appear in the metrics (round to 1 or 2 decimals), and say what they mean.
Then reply with one sentence giving the dashboard file path.""",
    tools=[build_dashboard],
    output_key="visualization_report",
    generate_content_config=GEN_CONFIG,
)

storage_agent = LlmAgent(
    name="storage_agent",
    model=config.MODEL,
    description="Uploads raw data, processed data and the dashboard to Google Cloud Storage.",
    instruction="""You are the storage step of a weather data pipeline.
Call `upload_to_cloud_storage` exactly once.
Then write the final pipeline report for the user in 3-5 short lines:
  - the bucket location and number of files uploaded (or why the upload was skipped/failed,
    including the hint if one was returned),
  - where the dashboard is: {visualization_report?}
  - the main finding in one sentence.""",
    tools=[upload_to_cloud_storage],
    output_key="storage_report",
    generate_content_config=GEN_CONFIG,
)

# ADK 2.x marks SequentialAgent as deprecated in favour of the newer graph-based
# Workflow API, but it remains fully supported and is the documented pattern for a
# fixed-order multi-agent pipeline. (Migrating is listed in the improvements doc.)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="SequentialAgent is deprecated")
    root_agent = SequentialAgent(
        name="hk_weather_pipeline",
        description="End-to-end agentic data pipeline: ingest -> process -> visualize -> store.",
        sub_agents=[ingestion_agent, processing_agent, visualization_agent, storage_agent],
    )
