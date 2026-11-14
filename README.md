# Hong Kong Weather — Agentic Data Pipeline (Google ADK)

An end-to-end data engineering pipeline in which **AI agents built with Google's Agent
Development Kit (ADK)** ingest open data from the **Hong Kong Observatory (HKO)**, process
it, publish an HTML dashboard, and store all raw and processed data in **Google Cloud Storage**.


## What it does

| Step | Agent | Tool | Output |
|---|---|---|---|
| 1. Ingestion | `ingestion_agent` | `fetch_hko_data` | Raw JSON: 140+ years of daily mean temperature (HKO `CLMTEMP`) and the current 9-day forecast (`fnd`) |
| 2. Processing | `processing_agent` | `transform_weather_data` | Cleaned daily data, annual summary, monthly comparison, enriched forecast, `metrics.json` |
| 3. Visualization | `visualization_agent` | `build_dashboard` | `dashboard.html` with charts, tables and an agent-written summary |
| 4. Storage | `storage_agent` | `upload_to_cloud_storage` | All files uploaded to `gs://<bucket>/hk-weather/<run_id>/` |

### Why it is agentic (not one script)

- A **`SequentialAgent`** (`hk_weather_pipeline`) orchestrates four **`LlmAgent`s**, each
  responsible for one stage and owning one tool.
- Agents **decide the tool arguments from a plain-English request**. For example,
  *"Use King's Park station with a 1981–2010 baseline"* makes the ingestion agent pick
  station `KP` and the processing agent pass `baseline_start=1981, baseline_end=2010`.
- Stages **share data through ADK session state** (`run_dir`, file paths, metrics), so only
  small summaries pass through the LLM, never the 50,000-row dataset.
- The **visualization agent writes the dashboard's headline and insights** from the computed
  metrics (instructed to use only numbers that are in the metrics).
- Every tool returns a `status`. Agents **check results and report errors** (e.g. a bad
  station code or a failed upload) instead of continuing silently, and downstream tools
  refuse to run if an earlier step did not succeed.

### Processing logic

- **Cleaning / filtering:** drops days with missing values (`***`), records not flagged
  complete (`C`), implausible values (outside 0–40 °C), invalid dates and duplicates.
  A data-quality report counts every dropped row.
- **Grouping / aggregation:**
  - **By year:** annual mean, min and max daily mean, number of complete days. Years with
    fewer than 330 complete days (e.g. the current partial year, or 1940–1946 when records
    were interrupted) are excluded.
  - **Against a climate baseline (default 1991–2020):** anomaly of every year, plus a
    10-year rolling mean.
  - **Trend:** linear-regression warming trend in °C per decade (full record and since 1991).
  - **By month:** baseline monthly normals vs. the current year's monthly means.
  - **Forecast enrichment:** daily temperature range, a "rain likely" flag from the
    probability of significant rain, and each day's forecast vs. the monthly normal.

## Project structure

```
hk-weather-adk-pipeline/
├── hk_weather_agent/          # ADK agent package (what `adk web` / `adk run` load)
│   ├── agent.py               # root_agent: SequentialAgent of 4 LlmAgents
│   ├── ingest.py              # tool: HKO API → raw JSON
│   ├── transform.py           # tool + pure functions: cleaning & aggregation (pandas)
│   ├── visualize.py           # tool: HTML dashboard (Chart.js)
│   ├── storage.py             # tool: upload to Google Cloud Storage
│   └── config.py              # settings read from .env
├── tests/                     # unit tests + offline end-to-end agent test
├── run_pipeline.py            # command-line runner
├── requirements.txt
├── .env.example               # template for your own .env (never commit .env)
└── README.md
```

## Setup

**Prerequisites:** Python 3.10+, a free [Gemini API key](https://aistudio.google.com),
a Google Cloud project with a Cloud Storage bucket, and the
[Google Cloud CLI](https://cloud.google.com/sdk/docs/install).

```bash
# 1. Clone and enter the project
git clone <this-repo-url>
cd hk-weather-adk-pipeline

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Add your settings
cp .env.example .env                 # Windows: copy .env.example .env
#    then edit .env: set GOOGLE_API_KEY, GOOGLE_CLOUD_PROJECT and GCS_BUCKET

# 4. Log in to Google Cloud (lets the code upload to your bucket; no key file needed)
gcloud auth application-default login
gcloud config set project <your-project-id>
```

The HKO API is public and needs no key.

## Run

**Option A — command line (recommended for grading):**

```bash
python run_pipeline.py
```

or with a custom request:

```bash
python run_pipeline.py "Use King's Park station (KP) with a 1981-2010 baseline"
```

The console shows each agent's tool call and reply, then the dashboard path. Open
`output/dashboard.html` in a browser.

**Option B — ADK developer UI** (watch each agent step by step):

```bash
adk web
```

Open the URL it prints (usually http://localhost:8000), choose `hk_weather_agent`, and type
a request such as *"Run the Hong Kong weather pipeline."* (`adk run hk_weather_agent` gives
the same thing in the terminal.)

## Outputs

Each run gets its own id (UTC timestamp) and folder:

```
output/runs/<run_id>/
├── raw/clmtemp_<station>.json         # untouched API response
├── raw/forecast_9day.json
├── processed/daily_clean.csv
├── processed/annual_summary.csv
├── processed/monthly_comparison.csv
├── processed/forecast_9day.csv
├── processed/metrics.json
└── dashboard.html
```

The same tree is uploaded to `gs://<GCS_BUCKET>/<GCS_PREFIX>/<run_id>/`. A copy of the
latest dashboard is also kept at `output/dashboard.html`.

If `GCS_BUCKET` is left empty, the pipeline still runs end to end and the storage agent
reports that the upload was skipped. That way the rest of the pipeline can be checked
without a cloud account.

## Tests

```bash
pytest -q
```

- `tests/test_transform.py` covers the cleaning and aggregation logic on synthetic data.
- `tests/test_pipeline_offline.py` runs the **full ADK agent workflow** with a scripted
  stand-in for Gemini and a mocked API. It checks that all four agents run in order and
  share state correctly, that LLM-written text is HTML-escaped, and that the storage tool
  uploads every file to the right GCS paths. No network access or API key is needed.

## Security notes

- No credentials are in the repository. The Gemini key lives in `.env`, which is in
  `.gitignore`, and Google Cloud access uses Application Default Credentials from
  `gcloud auth application-default login`.
- The bucket stays private (public access prevention on). The dashboard is opened locally,
  or shared from the bucket with signed URLs if needed.

## Notes

- Free-tier Gemini keys are rate-limited. The agents retry automatically with exponential
  backoff on HTTP 429 or 5xx errors, so a run can pause for a few seconds.
- The model can be changed with `GEMINI_MODEL` in `.env` (default `gemini-3.5-flash-lite`).
- Data source: [Hong Kong Observatory Open Data](https://www.hko.gov.hk/en/abouthko/opendata_intro.htm).
  The HKO publishes this data for free use; check its terms for attribution requirements.
