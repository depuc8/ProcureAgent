#!/usr/bin/env bash
# Build and deploy the public API to Cloud Run.
#
#   PROJECT_ID=my-proj REGION=us-central1 \
#   AGENT_ENGINE_RESOURCE_NAME=projects/.../locations/.../reasoningEngines/123 ./deploy/deploy_cloud_run.sh
#
# The service is reachable on the internet (--allow-unauthenticated at the IAM layer) because the
# app enforces its own bearer token (PROCURE_API_TOKEN, mounted from Secret Manager). It fails
# closed if the secret is missing.
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
: "${AGENT_ENGINE_RESOURCE_NAME:?set AGENT_ENGINE_RESOURCE_NAME (printed by deploy.agent_engine)}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-procureagent-api}"

gcloud run deploy "$SERVICE" \
  --project "$PROJECT_ID" --region "$REGION" --source . \
  --service-account "procure-api@${PROJECT_ID}.iam.gserviceaccount.com" \
  --allow-unauthenticated \
  --set-env-vars "AGENT_BACKEND=agent_engine,AGENT_ENGINE_RESOURCE_NAME=${AGENT_ENGINE_RESOURCE_NAME},GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${REGION},GOOGLE_GENAI_USE_VERTEXAI=TRUE" \
  --set-secrets "PROCURE_API_TOKEN=procure-api-token:latest" \
  --min-instances 0 --max-instances 5 --concurrency 20 --timeout 300

URL=$(gcloud run services describe "$SERVICE" --project "$PROJECT_ID" --region "$REGION" --format 'value(status.url)')
echo "deployed: $URL"
echo "try: curl -s $URL/healthz"
