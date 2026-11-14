"""Step 3 - Visualization: render the processed data as a single HTML dashboard."""

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from google.adk.tools import ToolContext

from . import config

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _fmt(value, suffix="", signed=False):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{value:+.2f}{suffix}" if signed else f"{value:.2f}{suffix}"


def _none_if_nan(series: pd.Series) -> list:
    return [None if pd.isna(v) else round(float(v), 2) for v in series]


def render_dashboard(
    annual: pd.DataFrame,
    monthly: pd.DataFrame,
    forecast: pd.DataFrame,
    metrics: dict,
    headline: str,
    insights: list[str],
    run_id: str,
    storage_location: str,
) -> str:
    """Build the dashboard HTML (Chart.js from a CDN, data embedded as JSON)."""
    esc = html.escape
    # Reindex to every calendar year so gaps (e.g. 1940-1946) show as breaks, not straight lines.
    annual = annual.set_index("year").reindex(range(int(annual["year"].min()), int(annual["year"].max()) + 1))
    annual = annual.rename_axis("year").reset_index()
    chart_data = {
        "years": annual["year"].astype(int).tolist(),
        "annualMean": _none_if_nan(annual["mean_temp"]),
        "rolling": _none_if_nan(annual["rolling_10yr_mean"]),
        "months": MONTHS,
        "normal": _none_if_nan(monthly["baseline_normal"]),
        "latest": _none_if_nan(monthly["latest_year_mean"]),
    }

    insight_items = "\n".join(f"<li>{esc(str(i))}</li>" for i in insights if str(i).strip())

    forecast_rows = ""
    for r in forecast.itertuples():
        rain = "Likely" if r.rain_likely else "Unlikely"
        forecast_rows += (
            f"<tr><td>{pd.Timestamp(r.date):%a %d %b}</td>"
            f"<td class='num'>{r.min_temp:.0f}&ndash;{r.max_temp:.0f}&nbsp;&deg;C</td>"
            f"<td class='num'>{_fmt(r.vs_monthly_normal, ' &deg;C', signed=True)}</td>"
            f"<td>{esc(str(r.rain_probability))} <span class='muted'>({rain})</span></td>"
            f"<td class='wx'>{esc(str(r.weather))}</td></tr>"
        )
    if not forecast_rows:
        forecast_rows = "<tr><td colspan='5'>No forecast available.</td></tr>"

    warm_rows = "".join(
        f"<tr><td>{w['year']}</td><td class='num'>{w['mean_temp_c']:.2f} &deg;C</td></tr>"
        for w in metrics["warmest_5_years"]
    )
    q = metrics["data_quality"]
    quality_rows = "".join(
        f"<tr><td>{esc(k.replace('_', ' ').capitalize())}</td><td class='num'>{v:,}</td></tr>"
        for k, v in q.items()
    )

    kpis = [
        ("Warming trend, full record", _fmt(metrics["warming_trend_c_per_decade_full_record"], " °C", True), "per decade, " + metrics["record_years"]),
        ("Warming trend since 1991", _fmt(metrics["warming_trend_c_per_decade_since_1991"], " °C", True), "per decade"),
        (f"{metrics['latest_full_year']} vs normal", _fmt(metrics["latest_full_year_anomaly_c"], " °C", True), f"vs {metrics['baseline_period']} average"),
        ("Rain likely, next 9 days", f"{metrics['forecast_rain_likely_days']} of {metrics['forecast_days']}", "days with Medium High or High chance"),
    ]
    kpi_html = "".join(
        f"<div class='kpi'><div class='kpi-label'>{esc(a)}</div><div class='kpi-value'>{esc(b)}</div>"
        f"<div class='kpi-sub'>{esc(c)}</div></div>"
        for a, b, c in kpis
    )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hong Kong Temperature Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --surface: #fcfcfb; --card: #ffffff; --border: #e4e3de;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #7a7973;
    --series-1: #2a78d6; --series-2: #eb6834; --reference: #b9b8b1;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--surface); color: var(--text-primary);
         font: 15px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px 48px; }}
  h1 {{ font-size: 26px; margin: 0 0 4px; }}
  h2 {{ font-size: 17px; margin: 0 0 4px; }}
  .meta, .muted, .kpi-sub, .note {{ color: var(--text-secondary); font-size: 13px; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px;
          padding: 20px; margin-top: 20px; }}
  .summary h2 {{ font-size: 20px; }}
  .summary ul {{ margin: 8px 0 0; padding-left: 20px; }}
  .tag {{ display: inline-block; font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
         color: var(--text-secondary); border: 1px solid var(--border); border-radius: 999px;
         padding: 2px 8px; margin-bottom: 8px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 20px; }}
  .kpi {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }}
  .kpi-label {{ font-size: 13px; color: var(--text-secondary); }}
  .kpi-value {{ font-size: 28px; font-weight: 600; margin: 2px 0; font-variant-numeric: tabular-nums; }}
  .chart-wrap {{ position: relative; height: 340px; margin-top: 12px; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 760px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }}
  th, td {{ text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--border); vertical-align: top; }}
  th {{ color: var(--text-secondary); font-weight: 600; font-size: 13px; }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
  td.wx {{ color: var(--text-secondary); font-size: 13px; }}
  footer {{ margin-top: 28px; color: var(--text-muted); font-size: 12px; }}
  code {{ background: #f1f0ec; padding: 1px 5px; border-radius: 4px; }}
</style>
</head>
<body>
<main>
  <h1>Hong Kong Temperature Dashboard</h1>
  <div class="meta">Station: {esc(metrics['station'])} &middot; Records {esc(metrics['record_years'])}
    &middot; Baseline {esc(metrics['baseline_period'])} (annual mean {metrics['baseline_annual_mean_c']:.2f} &deg;C)
    &middot; Run <code>{esc(run_id)}</code> &middot; Generated {generated}</div>

  <section class="card summary">
    <span class="tag">Summary written by the visualization agent</span>
    <h2>{esc(headline)}</h2>
    <ul>{insight_items}</ul>
  </section>

  <section class="kpis">{kpi_html}</section>

  <section class="card">
    <h2>Annual mean temperature</h2>
    <div class="note">Years with fewer than the minimum number of complete daily records (including the current partial year) are excluded.</div>
    <div class="chart-wrap"><canvas id="annualChart" aria-label="Annual mean temperature line chart"></canvas></div>
  </section>

  <section class="card">
    <h2>{metrics['current_year']} monthly mean vs. {esc(metrics['baseline_period'])} normal</h2>
    <div class="note">Months with fewer than 15 complete days in {metrics['current_year']} are left blank.</div>
    <div class="chart-wrap"><canvas id="monthlyChart" aria-label="Monthly comparison bar chart"></canvas></div>
  </section>

  <section class="card">
    <h2>9-day forecast</h2>
    <div class="note">"vs normal" compares each day's forecast mean ((max + min) / 2) with the baseline normal for that month.</div>
    <table>
      <thead><tr><th>Date</th><th class="num">Temperature</th><th class="num">vs normal</th><th>Chance of significant rain</th><th>Outlook</th></tr></thead>
      <tbody>{forecast_rows}</tbody>
    </table>
  </section>

  <div class="grid2">
    <section class="card">
      <h2>Five warmest years on record</h2>
      <table><thead><tr><th>Year</th><th class="num">Annual mean</th></tr></thead><tbody>{warm_rows}</tbody></table>
    </section>
    <section class="card">
      <h2>Data quality report</h2>
      <table><thead><tr><th>Check</th><th class="num">Rows</th></tr></thead><tbody>{quality_rows}</tbody></table>
    </section>
  </div>

  <footer>
    Source: Hong Kong Observatory Open Data API (daily mean temperature, CLMTEMP; 9-day forecast, fnd).
    Raw and processed files: {esc(storage_location)}.
    Built by an agentic pipeline using Google's Agent Development Kit (ADK).
  </footer>
</main>
<script>
const D = {json.dumps(chart_data)};
const css = getComputedStyle(document.documentElement);
const c = n => css.getPropertyValue(n).trim();
Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
Chart.defaults.color = c('--text-secondary');
const grid = {{ color: c('--border') }};

new Chart(document.getElementById('annualChart'), {{
  type: 'line',
  data: {{ labels: D.years, datasets: [
    {{ label: 'Annual mean (°C)', data: D.annualMean, borderColor: c('--series-1'), backgroundColor: c('--series-1'),
       borderWidth: 1.5, pointRadius: 0, pointHoverRadius: 4, spanGaps: false }},
    {{ label: '10-year rolling mean (°C)', data: D.rolling, borderColor: c('--series-2'), backgroundColor: c('--series-2'),
       borderWidth: 2.5, pointRadius: 0, pointHoverRadius: 4, spanGaps: false }}
  ]}},
  options: {{ maintainAspectRatio: false, interaction: {{ mode: 'index', intersect: false }},
    plugins: {{ legend: {{ position: 'top', align: 'start' }},
      tooltip: {{ callbacks: {{ label: ctx => `${{ctx.dataset.label}}: ${{ctx.parsed.y?.toFixed(2)}}` }} }} }},
    scales: {{ x: {{ grid: {{ display: false }}, ticks: {{ maxTicksLimit: 14 }} }},
              y: {{ grid, title: {{ display: true, text: '°C' }} }} }} }}
}});

new Chart(document.getElementById('monthlyChart'), {{
  type: 'bar',
  data: {{ labels: D.months, datasets: [
    {{ label: 'Baseline normal (°C)', data: D.normal, backgroundColor: c('--reference'), borderRadius: 4, borderSkipped: 'bottom' }},
    {{ label: '{metrics['current_year']} mean (°C)', data: D.latest, backgroundColor: c('--series-1'), borderRadius: 4, borderSkipped: 'bottom' }}
  ]}},
  options: {{ maintainAspectRatio: false, interaction: {{ mode: 'index', intersect: false }},
    datasets: {{ bar: {{ categoryPercentage: 0.7, barPercentage: 0.9 }} }},
    plugins: {{ legend: {{ position: 'top', align: 'start' }},
      tooltip: {{ callbacks: {{
        label: ctx => ctx.parsed.y == null ? `${{ctx.dataset.label}}: no data` : `${{ctx.dataset.label}}: ${{ctx.parsed.y.toFixed(2)}}`,
        footer: items => {{ const n = items[0]?.chart.data.datasets[0].data[items[0].dataIndex];
          const l = items[0]?.chart.data.datasets[1].data[items[0].dataIndex];
          return (n != null && l != null) ? `Difference: ${{(l - n >= 0 ? '+' : '')}}${{(l - n).toFixed(2)}} °C` : ''; }} }} }} }},
    scales: {{ x: {{ grid: {{ display: false }} }}, y: {{ grid, beginAtZero: true, title: {{ display: true, text: '°C' }} }} }} }}
}});
</script>
</body>
</html>
"""


def build_dashboard(headline: str, insights: list[str], tool_context: ToolContext) -> dict:
    """Creates the HTML dashboard from the processed data.

    Args:
        headline: One short headline (max ~15 words) summarising the main finding.
        insights: 3 to 4 short, factual bullet points written from the metrics.
            Only use numbers that appear in the metrics.

    Returns:
        Status and the local path of the dashboard file.
    """
    files = tool_context.state.get("processed_files")
    run_dir = tool_context.state.get("run_dir")
    if not files or not run_dir:
        return {"status": "error", "message": "No processed data found - the processing step must succeed first."}

    annual = pd.read_csv(files["annual_summary"])
    monthly = pd.read_csv(files["monthly_comparison"])
    forecast = pd.read_csv(files["forecast_9day"], parse_dates=["date"])
    metrics = json.loads(Path(files["metrics"]).read_text(encoding="utf-8"))
    run_id = tool_context.state.get("run_id", "")

    if config.GCS_BUCKET:
        location = f"gs://{config.GCS_BUCKET}/{config.GCS_PREFIX}/{run_id}/"
    else:
        location = "saved locally (no GCS_BUCKET configured)"

    page = render_dashboard(annual, monthly, forecast, metrics, headline, insights or [], run_id, location)
    path = Path(run_dir) / "dashboard.html"
    path.write_text(page, encoding="utf-8")

    # Convenience copy so the latest result is always at output/dashboard.html
    latest = config.OUTPUT_DIR / "dashboard.html"
    latest.write_text(page, encoding="utf-8")

    tool_context.state["dashboard_path"] = str(path)
    return {"status": "success", "dashboard_path": str(path), "latest_copy": str(latest)}
