# Cloud Run image for the public API (api/main.py). Fronts the Agent Engine deployment by default.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 AGENT_BACKEND=agent_engine

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY settings.py ./
COPY agents ./agents
COPY tools ./tools
COPY observability ./observability
COPY data ./data
COPY api ./api

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

# Cloud Run injects PORT
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
