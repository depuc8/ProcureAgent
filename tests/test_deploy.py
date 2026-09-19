import sys

import cloudpickle
import pytest

import settings
from deploy import agent_engine as ae


@pytest.fixture
def args():
    return ae.argparse.Namespace(
        display_name="procureagent", model="gemini-3.8-flash", bq_dataset="procureagent", bq_location="US",
        service_account="sa@p.iam.gserviceaccount.com", staging_bucket=None, python_version="3.12",
        min_instances=0, max_instances=5)


def test_config_is_valid_for_the_installed_sdk(args):
    """Catches a mistyped / removed config key at test time instead of deploy time."""
    from agentplatform._genai.types.common import AgentRuntimeConfig

    config = ae.build_config(args)
    parsed = AgentRuntimeConfig(**config)
    assert parsed.agent_framework == "google-adk" and parsed.python_version == "3.12"
    assert not set(config["env_vars"]) & ae.RESERVED_ENV


def test_requirements_are_pinned_to_local_versions(args):
    reqs = ae.build_config(args)["requirements"]
    assert all("==" in r for r in reqs)
    assert any(r.startswith("google-adk==") for r in reqs)
    assert any(r.startswith("google-cloud-aiplatform[agent_engines,adk]==") for r in reqs)


def test_every_shipped_package_exists():
    for path in ae.EXTRA_PACKAGES:
        assert (ae.ROOT / path).exists(), path
    assert (ae.ROOT / "data" / "policy.json").exists() and len(list((ae.ROOT / "data" / "contracts").glob("*.txt"))) == 10


def test_adk_app_pickles_and_round_trips():
    """Agent Engine cloudpickles the AdkApp; an unpicklable member (lock, client) breaks deploys."""
    import vertexai

    vertexai.init(project="test-project", location="us-central1")
    app = ae.build_adk_app()
    clone = cloudpickle.loads(cloudpickle.dumps(app))
    assert clone._tmpl_attrs["app"].root_agent.name == "procure_planner"


def test_dry_run_deploys_nothing(capsys):
    assert ae.main(["--project", "test-project", "--dry-run"]) == 0
    assert "nothing was deployed" in capsys.readouterr().out


def test_python_version_mismatch_is_refused():
    local = f"{sys.version_info.major}.{sys.version_info.minor}"
    other = "3.11" if local != "3.11" else "3.12"
    with pytest.raises(SystemExit, match="cloudpickle"):
        ae.check_python(other, allow_mismatch=False)
    ae.check_python(local, allow_mismatch=False)
    ae.check_python(other, allow_mismatch=True)


# ---- secrets ----------------------------------------------------------------------------------
def test_secret_prefers_environment(monkeypatch):
    monkeypatch.setenv("PROCURE_API_TOKEN", "from-env")
    monkeypatch.setenv("SECRET_MANAGER_ENABLED", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    assert settings.get_secret("PROCURE_API_TOKEN") == "from-env"


def test_secret_falls_back_to_secret_manager_with_derived_id(monkeypatch):
    seen = {}

    def fake(project, secret_id):
        seen.update(project=project, secret_id=secret_id)
        return "from-sm"

    monkeypatch.delenv("PROCURE_API_TOKEN", raising=False)
    monkeypatch.setenv("SECRET_MANAGER_ENABLED", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-proj")
    monkeypatch.setattr(settings, "_secret_from_manager", fake)
    assert settings.get_secret("PROCURE_API_TOKEN") == "from-sm"
    assert seen == {"project": "my-proj", "secret_id": "procure-api-token"}


def test_secret_absent_when_disabled(monkeypatch):
    monkeypatch.delenv("PROCURE_API_TOKEN", raising=False)
    monkeypatch.setenv("SECRET_MANAGER_ENABLED", "false")
    assert settings.get_secret("PROCURE_API_TOKEN") is None


def test_no_credentials_in_repo():
    import re
    from pathlib import Path

    pat = re.compile(r"AIza[0-9A-Za-z_\-]{35}|-----BEGIN (RSA |EC )?PRIVATE KEY-----|ya29\.[0-9A-Za-z_\-]{20,}")
    for f in ae.ROOT.rglob("*"):
        if f.is_file() and f.suffix in {".py", ".json", ".txt", ".md", ".sh", ".yaml", ".toml", ".example"} \
                and ".venv" not in f.parts and ".git" not in f.parts:
            assert not pat.search(f.read_text(errors="ignore")), f
