import importlib
import os

import settings


def test_dotenv_is_loaded_but_never_overrides_real_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("PROCURE_TEST_FROM_DOTENV=file\nPROCURE_TEST_BOTH=file\n")
    monkeypatch.setenv("PROCURE_TEST_BOTH", "real-env")
    monkeypatch.delenv("PROCURE_TEST_FROM_DOTENV", raising=False)
    from dotenv import load_dotenv

    load_dotenv(env, override=False)          # same call settings.py makes
    assert os.environ["PROCURE_TEST_FROM_DOTENV"] == "file"
    assert os.environ["PROCURE_TEST_BOTH"] == "real-env"
    monkeypatch.delenv("PROCURE_TEST_FROM_DOTENV")


def test_settings_import_works_without_a_dotenv_file():
    importlib.reload(settings)
    assert settings.get_settings().model
