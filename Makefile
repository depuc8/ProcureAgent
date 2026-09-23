PY ?= .venv/bin/python
ADK ?= .venv/bin/adk

.PHONY: install test test-live web seed seed-dry dashboard api deploy-dry-run benchmark benchmark-live contracts

install:
	python3 -m venv .venv && .venv/bin/pip install -e ".[dashboard,dev]"

test:
	$(PY) -m pytest -q

test-live:
	PROCURE_RUN_LIVE=1 $(PY) -m pytest -q -m live

web:                       # ADK dev UI; discovers agents/ (run from the repo root)
	$(ADK) web .

seed:
	$(PY) -m data.seed_bigquery

seed-dry:
	$(PY) -m data.seed_bigquery --dry-run

contracts:                 # regenerate data/contracts/*.txt
	$(PY) -m data.generate_contracts

dashboard:
	$(PY) -m streamlit run dashboard/app.py

api:                       # local dev server; in-process agents, no auth (dev override)
	AGENT_BACKEND=local PROCURE_API_ALLOW_UNAUTHENTICATED=true $(PY) -m uvicorn api.main:app --reload --port 8080

deploy-dry-run:
	$(PY) -m deploy.agent_engine --project $${GOOGLE_CLOUD_PROJECT:-your-project} --dry-run

benchmark:
	$(PY) -m scripts.benchmark

benchmark-live:
	PROCURE_RUN_LIVE=1 $(PY) -m scripts.benchmark --live --runs 5
