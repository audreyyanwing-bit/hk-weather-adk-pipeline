"""End-to-end test of the ADK agent workflow without network access or an API key.

A scripted fake LLM stands in for Gemini: for each agent it calls that agent's
tool once and then replies with text. The HKO API is replaced by fixtures.
This proves the orchestration (SequentialAgent, tools, shared session state)
works; the real run uses Gemini.
"""

import asyncio
from typing import AsyncGenerator

from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.models import LlmCapabilities
from google.adk.runners import InMemoryRunner
from google.genai import types

from hk_weather_agent import agent as pipeline
from hk_weather_agent import ingest

from .conftest import make_clmtemp, make_forecast

TOOL_ARGS = {
    "fetch_hko_data": {"station": "HKO"},
    "transform_weather_data": {},
    "build_dashboard": {"headline": "Hong Kong keeps getting warmer",
                        "insights": ["Test insight one.", "Test insight two.", "Test <b>escaped</b>."]},
    "upload_to_cloud_storage": {},
}


class ScriptedLlm(BaseLlm):
    @property
    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities(output_schema_and_tools=True)

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        last = llm_request.contents[-1] if llm_request.contents else None
        if last and any(p.function_response for p in last.parts or []):
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text="Step done.")]))
            return
        tool_name = llm_request.config.tools[0].function_declarations[0].name
        yield LlmResponse(content=types.Content(role="model", parts=[
            types.Part(function_call=types.FunctionCall(name=tool_name, args=TOOL_ARGS[tool_name]))]))


def test_full_pipeline_offline(isolated_output, monkeypatch):
    fixtures = {"CLMTEMP": make_clmtemp(), "fnd": make_forecast()}
    monkeypatch.setattr(ingest, "_get_json", lambda url, params, attempts=3: fixtures[params["dataType"]])
    for sub in pipeline.root_agent.sub_agents:
        monkeypatch.setattr(sub, "model", ScriptedLlm(model="scripted"))

    async def run():
        runner = InMemoryRunner(agent=pipeline.root_agent, app_name="test")
        session = await runner.session_service.create_session(app_name="test", user_id="u")
        authors = []
        async for event in runner.run_async(
            user_id="u", session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text="Run the pipeline")]),
        ):
            authors.append(event.author)
        final = await runner.session_service.get_session(app_name="test", user_id="u", session_id=session.id)
        return authors, final.state

    authors, state = asyncio.run(run())

    # Every agent ran, in order
    order = [a for i, a in enumerate(authors) if a != "user" and a not in authors[:i]]
    assert order == ["ingestion_agent", "processing_agent", "visualization_agent", "storage_agent"]

    # Each step left its outputs in shared state
    assert state["station"] == "HKO"
    assert set(state["processed_files"]) >= {"annual_summary", "monthly_comparison", "forecast_9day", "metrics"}
    html = open(state["dashboard_path"], encoding="utf-8").read()
    assert "Hong Kong keeps getting warmer" in html
    assert "&lt;b&gt;escaped&lt;/b&gt;" in html          # LLM text is HTML-escaped
    assert "annualChart" in html and "monthlyChart" in html
    assert (isolated_output / "dashboard.html").exists()


def test_storage_uploads_every_file_to_gcs(isolated_output, monkeypatch):
    """The storage tool uploads raw/processed/dashboard files under <prefix>/<run_id>/."""
    import sys
    import types as pytypes

    from hk_weather_agent import config, storage

    run_dir = isolated_output / "runs" / "R1"
    for rel in ["raw/a.json", "processed/b.csv", "dashboard.html"]:
        (run_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (run_dir / rel).write_text("x")

    uploaded = []

    class FakeBlob:
        def __init__(self, name): self.name = name
        def upload_from_filename(self, filename, content_type=None): uploaded.append((self.name, content_type))

    class FakeClient:
        def bucket(self, name):
            return pytypes.SimpleNamespace(blob=FakeBlob)

    import google.cloud

    fake_storage = pytypes.SimpleNamespace(Client=FakeClient)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", fake_storage)
    monkeypatch.setattr(google.cloud, "storage", fake_storage, raising=False)
    monkeypatch.setattr(config, "GCS_BUCKET", "demo-bucket")

    ctx = pytypes.SimpleNamespace(state={"run_dir": str(run_dir), "run_id": "R1"})
    result = storage.upload_to_cloud_storage(ctx)

    assert result["status"] == "success" and result["uploaded_count"] == 3
    assert ("hk-weather/R1/processed/b.csv", "text/csv") in uploaded
    assert result["bucket_location"] == "gs://demo-bucket/hk-weather/R1/"
