"""Smoke-test a deployed Agent Engine instance.

    python -m deploy.query_agent --project p --resource projects/p/locations/l/reasoningEngines/123 \\
        "We need 50 laptops for new hires, budget ₹40L, delivery in 3 weeks"
"""

from __future__ import annotations

import argparse
import asyncio


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--location", default="us-central1")
    ap.add_argument("--resource", required=True)
    ap.add_argument("--user-id", default="smoke-test")
    ap.add_argument("--session-id")
    ap.add_argument("message")
    a = ap.parse_args()

    import agentplatform

    client = agentplatform.Client(project=a.project, location=a.location)
    remote = client.runtimes.get(name=a.resource)
    session_id = a.session_id or (await remote.async_create_session(user_id=a.user_id))["id"]
    print(f"session: {session_id}\n")
    async for event in remote.async_stream_query(user_id=a.user_id, session_id=session_id, message=a.message):
        for part in (event.get("content") or {}).get("parts", []):
            if part.get("text") and not event.get("partial"):
                print(part["text"])


if __name__ == "__main__":
    asyncio.run(main())
