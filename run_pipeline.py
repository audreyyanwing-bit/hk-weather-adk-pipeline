"""Run the whole agentic pipeline from the command line.

Usage:
    python run_pipeline.py
    python run_pipeline.py "Use King's Park station with a 1981-2010 baseline"

Prints each agent's tool calls and replies as they happen, then the final report.
"""

import asyncio
import json
import sys

from google.adk.runners import InMemoryRunner
from google.genai import types

from hk_weather_agent.agent import root_agent

DEFAULT_REQUEST = (
    "Run the Hong Kong weather pipeline for the Hong Kong Observatory headquarters "
    "station, using the 1991-2020 baseline."
)


def _short(obj, limit=300) -> str:
    text = json.dumps(obj, default=str) if not isinstance(obj, str) else obj
    return text if len(text) <= limit else text[:limit] + " ..."


async def main(request: str) -> int:
    runner = InMemoryRunner(agent=root_agent, app_name="hk_weather_pipeline")
    session = await runner.session_service.create_session(
        app_name="hk_weather_pipeline", user_id="local_user"
    )
    print(f"\nRequest: {request}\n" + "-" * 70)

    message = types.Content(role="user", parts=[types.Part(text=request)])
    async for event in runner.run_async(
        user_id="local_user", session_id=session.id, new_message=message
    ):
        if not event.content or not event.content.parts:
            continue
        for part in event.content.parts:
            if part.function_call:
                print(f"[{event.author}] -> calls {part.function_call.name}({_short(part.function_call.args)})")
            elif part.function_response:
                status = (part.function_response.response or {}).get("status", "")
                print(f"[{event.author}] <- {part.function_response.name} returned status={status}")
            elif part.text and part.text.strip():
                print(f"[{event.author}] {part.text.strip()}\n")

    final = await runner.session_service.get_session(
        app_name="hk_weather_pipeline", user_id="local_user", session_id=session.id
    )
    state = final.state if final else {}
    print("-" * 70)
    if state.get("dashboard_path"):
        print(f"Dashboard: {state['dashboard_path']}")
    if state.get("gcs_uris"):
        print(f"Uploaded {len(state['gcs_uris'])} files to Google Cloud Storage.")
    return 0 if state.get("dashboard_path") else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(" ".join(sys.argv[1:]) or DEFAULT_REQUEST)))
