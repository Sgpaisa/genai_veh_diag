# Vehicle Diagnostics GenAI Platform

A production-grade GenAI system built on Google Cloud Platform that autonomously diagnoses vehicle fleet errors using Gemini AI.

## Live API
```
https://vehicle-diagnostics-api-746368083201.asia-south1.run.app/docs
```

## Architecture
Cloud Storage → ETL (Python) → BigQuery → Gemini 2.5 Flash → Google ADK Agent → FastAPI on Cloud Run

## Stack
- **AI**: Gemini 2.5 Flash with response_schema structured output
- **Agent**: Google ADK (LlmAgent, FunctionTool, 4 autonomous tools)
- **Data**: BigQuery (partitioned by day, clustered by vehicle_id + error_code)
- **Search**: Vertex AI Vector Search + text-embedding-004
- **API**: FastAPI deployed on Cloud Run (auto-scaling, HTTPS)
- **Storage**: Google Cloud Storage (raw/ → processed/)

## API Endpoints
- `GET /health` — service status
- `GET /vehicle/{id}` — vehicle error history from BigQuery
- `GET /fleet/summary` — top error codes across fleet
- `GET /diagnose/{id}/{code}` — Gemini AI diagnosis with risk level
- `GET /search?symptom=` — semantic search via Vertex AI Vector Search

## Key Features
- Structured JSON output via Pydantic + response_schema (zero regex parsing)
- Autonomous agent triages fleet — High risk triggers alert, Medium/Low monitored
- Semantic search finds similar errors by symptom description (87% cosine similarity)
- Full data lineage (source_file + processed_at on every BigQuery row)
- IAM least-privilege service account (4 specific roles only)
