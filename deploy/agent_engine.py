"""Deploy ProcureAgent to Vertex AI Agent Engine (Agent Runtime).

    python -m deploy.agent_engine --project my-proj --location us-central1 --dry-run
    python -m deploy.agent_engine --project my-proj --location us-central1 \\
        --service-account procure-agent@my-proj.iam.gserviceaccount.com
    python -m deploy.agent_engine --project my-proj --update projects/.../reasoningEngines/123

Object-based deployment: the AdkApp is cloudpickled and shipped with the packages in
EXTRA_PACKAGES. cloudpickle needs the *same Python minor version* locally and remotely, so
build the deploy environment with the version you pass as --python-version.

--dry-run builds the app, checks it pickles, prints the exact config and exits without
touching GCP.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTRA_PACKAGES = ["agents", "tools", "observability", "data", "settings.py"]
# distributions pinned to the *local* versions so the server unpickles with the code that pickled it
PINNED = ["google-adk", "google-cloud-aiplatform", "google-cloud-bigquery", "google-genai",
          "cloudpickle", "pydantic", "opentelemetry-exporter-otlp-proto-http"]
AGENT_ENGINE_ENV = {
    "GOOGLE_GENAI_USE_VERTEXAI": "TRUE",
    "VENDOR_BACKEND": "bigquery",
    "METRICS_TO_BIGQUERY": "true",
    "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY": "true",
}
# the runtime owns these; setting them is rejected or ignored
RESERVED_ENV = {"GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "PORT", "GOOGLE_APPLICATION_CREDENTIALS"}


def pinned_requirements() -> list[str]:
    reqs = []
    for dist in PINNED:
        try:
            version = metadata.version(dist)
        except metadata.PackageNotFoundError:
            raise SystemExit(f"{dist} is not installed in this environment; install requirements.txt first.")
        extras = "[agent_engines,adk]" if dist == "google-cloud-aiplatform" else ""
        reqs.append(f"{dist}{extras}=={version}")
    return reqs


def build_config(args: argparse.Namespace) -> dict:
    env = dict(AGENT_ENGINE_ENV)
    env["PROCURE_MODEL"] = args.model
    env["BQ_DATASET"] = args.bq_dataset
    env["BQ_LOCATION"] = args.bq_location
    bad = RESERVED_ENV & set(env)
    assert not bad, f"reserved env vars set: {bad}"
    config: dict = {
        "display_name": args.display_name,
        "description": "ProcureAgent: multi-agent procurement assistant (ADK + Gemini)",
        "requirements": pinned_requirements(),
        "extra_packages": EXTRA_PACKAGES,
        "env_vars": env,
        "agent_framework": "google-adk",
        "python_version": args.python_version,
        "min_instances": args.min_instances,
        "max_instances": args.max_instances,
        "resource_limits": {"cpu": "4", "memory": "8Gi"},
        "container_concurrency": 9,
        "labels": {"app": "procureagent"},
    }
    if args.service_account:
        config["service_account"] = args.service_account
    if args.staging_bucket:
        config["staging_bucket"] = args.staging_bucket
    return config


def build_adk_app():
    from vertexai.agent_engines import AdkApp

    from agents.agent import app

    # enable_tracing=True: AdkApp wires the Cloud Trace exporter; TelemetryPlugin (already on `app`)
    # decorates those spans and writes per-request metrics to BigQuery.
    return AdkApp(app=app, enable_tracing=True)


def check_python(target: str, allow_mismatch: bool) -> None:
    local = f"{sys.version_info.major}.{sys.version_info.minor}"
    if local != target and not allow_mismatch:
        raise SystemExit(
            f"Local Python is {local} but the deployment targets {target}. cloudpickle requires the same "
            f"minor version on both sides. Re-run from a {target} virtualenv (e.g. `uv venv --python {target}`), "
            f"or pass --python-version {local} if Agent Engine supports it, or --allow-python-mismatch to override."
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", required=True)
    p.add_argument("--location", default="us-central1")
    p.add_argument("--display-name", default="procureagent")
    p.add_argument("--model", default="gemini-3.8-flash")
    p.add_argument("--bq-dataset", default="procureagent")
    p.add_argument("--bq-location", default="US")
    p.add_argument("--service-account")
    p.add_argument("--staging-bucket", help="gs://bucket (optional; the SDK can manage staging itself)")
    p.add_argument("--python-version", default="3.12")
    p.add_argument("--allow-python-mismatch", action="store_true")
    p.add_argument("--min-instances", type=int, default=0)
    p.add_argument("--max-instances", type=int, default=5)
    p.add_argument("--update", metavar="RESOURCE_NAME", help="update this existing runtime instead of creating one")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    for path in EXTRA_PACKAGES:
        if not (ROOT / path).exists():
            raise SystemExit(f"extra package missing: {path}")

    import vertexai

    vertexai.init(project=args.project, location=args.location)
    adk_app = build_adk_app()

    import cloudpickle

    size = len(cloudpickle.dumps(adk_app))
    config = build_config(args)
    if args.dry_run:
        print(json.dumps({"project": args.project, "location": args.location, "pickled_bytes": size,
                          "update": args.update, "config": config}, indent=2))
        print("\ndry run OK: app builds and pickles; nothing was deployed.")
        return 0

    check_python(args.python_version, args.allow_python_mismatch)
    import agentplatform

    client = agentplatform.Client(project=args.project, location=args.location)
    if args.update:
        remote = client.runtimes.update(name=args.update, agent=adk_app, config=config)
    else:
        remote = client.runtimes.create(agent=adk_app, config=config)
    name = getattr(getattr(remote, "api_resource", remote), "name", None) or str(remote)
    print(f"Deployed: {name}")
    print("Set AGENT_ENGINE_RESOURCE_NAME to this value for the Cloud Run API.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
