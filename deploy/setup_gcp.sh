#!/usr/bin/env bash
# One-time GCP setup for ProcureAgent: APIs, service accounts + IAM, API token secret,
# BigQuery dataset/tables. Idempotent. Needs gcloud authenticated as a project admin.
#
#   PROJECT_ID=my-proj REGION=us-central1 ./deploy/setup_gcp.sh
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-us-central1}"
AGENT_SA="procure-agent@${PROJECT_ID}.iam.gserviceaccount.com"
API_SA="procure-api@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud config set project "$PROJECT_ID" >/dev/null

echo "== enabling APIs"
gcloud services enable aiplatform.googleapis.com bigquery.googleapis.com cloudtrace.googleapis.com \
  telemetry.googleapis.com logging.googleapis.com secretmanager.googleapis.com run.googleapis.com \
  cloudbuild.googleapis.com artifactregistry.googleapis.com

ensure_sa() { gcloud iam service-accounts describe "$1" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "${1%%@*}" --display-name "$2"; }

echo "== service accounts"
ensure_sa "$AGENT_SA" "ProcureAgent (Agent Engine runtime)"
ensure_sa "$API_SA" "ProcureAgent public API (Cloud Run)"

bind() { gcloud projects add-iam-policy-binding "$PROJECT_ID" --member "serviceAccount:$1" --role "$2" \
  --condition=None >/dev/null; }

echo "== IAM: agent runtime"
for role in roles/aiplatform.user roles/bigquery.dataViewer roles/bigquery.jobUser roles/bigquery.dataEditor \
            roles/cloudtrace.agent roles/logging.logWriter; do bind "$AGENT_SA" "$role"; done

echo "== IAM: API (calls Agent Engine, reads its API token)"
for role in roles/aiplatform.user roles/logging.logWriter; do bind "$API_SA" "$role"; done

echo "== API token secret (generated once; never printed)"
if ! gcloud secrets describe procure-api-token >/dev/null 2>&1; then
  openssl rand -base64 32 | tr -d '\n' | gcloud secrets create procure-api-token --data-file=- --replication-policy=automatic
  echo "created secret procure-api-token; read it with: gcloud secrets versions access latest --secret procure-api-token"
fi
gcloud secrets add-iam-policy-binding procure-api-token --member "serviceAccount:$API_SA" \
  --role roles/secretmanager.secretAccessor >/dev/null

echo "== BigQuery: dataset, vendor table (200 synthetic vendors), request_metrics table"
GOOGLE_CLOUD_PROJECT="$PROJECT_ID" python -m data.seed_bigquery

echo "done. Next: python -m deploy.agent_engine --project $PROJECT_ID --location $REGION --service-account $AGENT_SA"
