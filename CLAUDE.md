# Vehicle Diagnostics — Project Guide

## Architecture

```
Cloud Storage (raw/*.csv)
        │
        ▼
   etl.py  ──────────────────────────────────────────────────────────┐
   • reads CSVs, cleans, loads to BigQuery                           │
   • embeds error descriptions → Vertex AI Vector Search            │
        │                                                             │
        ▼                                                             ▼
BigQuery (error_logs)                                  Vertex AI Vector Search
        │                                              (semantic symptom search)
        ▼
diagnostics.py  ←─── Gemini 2.5 Flash (structured JSON output)
        │
        ▼
   api.py  (FastAPI, port 8080)
        │
        ▼
   agent.py  (Google ADK — primary autonomous orchestration layer)
```

Alternative framework implementations (for comparison only — not production):
`langchain_diagnose.py`, `langgraph_agent.py`, `fleet_crew.py`, `mcp_client.py` / `vehicle_mcp_server.py`

---

## GCP Stack

| Service | Purpose | Config key |
|---|---|---|
| Cloud Storage | Raw OBD-II CSV ingestion (`raw/`) and processed archive (`processed/`) | `BUCKET = "vehicle-logs-sachin"` |
| BigQuery | Fleet error log storage and querying | `BQ_TABLE = "vehicle-diagnostics-491610.vehicle_diagnostics.error_logs"` |
| Vertex AI — Gemini 2.5 Flash | Structured diagnosis with JSON output | `GEMINI_MODEL = "gemini-2.5-flash"` |
| Vertex AI — text-embedding-004 | 768-dim embeddings for vector search | `EMBEDDING_MODEL = "text-embedding-004"` |
| Vertex AI Vector Search | Semantic symptom-to-error-code lookup | `VS_INDEX_ID`, `VS_ENDPOINT_ID` in `config.py` |
| Cloud Run / Docker | API hosting | `Dockerfile`, `docker-compose.yml` |

Project: `vehicle-diagnostics-491610` | Region: `asia-south1`

All constants live in `config.py`. Never hardcode them elsewhere.

---

## BigQuery Schema

Table: `vehicle-diagnostics-491610.vehicle_diagnostics.error_logs`

| Column | Type | Description |
|---|---|---|
| `timestamp` | TIMESTAMP | OBD-II reading time (UTC) |
| `vehicle_id` | STRING | Vehicle identifier (e.g. `VEH101`) |
| `error_code` | STRING | OBD-II fault code (e.g. `P0171`) |
| `sensor` | STRING | Sensor name (e.g. `oxygen_sensor`) |
| `value` | FLOAT64 | Sensor reading |
| `unit` | STRING | Unit of measurement |
| `processed_at` | TIMESTAMP | ETL processing time |
| `source_file` | STRING | Source CSV filename |

**Query convention:** Always use named parameters (`@v`) with `QueryJobConfig` — never f-string interpolation for user-supplied values. See `diagnostics.py:get_vehicle_history` as the reference pattern.

---

## Primary Agent Framework

**Google ADK** (`agent.py`) is the production orchestration layer.

The agent is an `LlmAgent` (Gemini 2.5 Flash) with four `FunctionTool`s:

| Tool | When called |
|---|---|
| `fetch_and_load()` | Always first — pulls CSVs from GCS, loads to BigQuery, indexes embeddings |
| `fleet_summary()` | After load — queries BigQuery for per-vehicle error counts |
| `diagnose_error(vehicle_id, error_code, sensor, value)` | For every vehicle+error pair |
| `send_alert(vehicle_id, error_code, summary)` | Only when `risk_level == "High"` |

Run the agent:
```bash
python agent.py
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/vehicle/{vehicle_id}` | Last 20 errors for a vehicle |
| GET | `/fleet/summary` | Top 10 error codes across fleet |
| GET | `/diagnose/{vehicle_id}/{error_code}` | Gemini AI diagnosis (risk level + actions) |
| GET | `/search?symptom=...` | Semantic search via Vertex AI Vector Search |

---

## Coding Conventions

- **SQL:** Parameterized queries only. Use `bigquery.ScalarQueryParameter("v", "STRING", value)` with `@v` placeholders. No f-string interpolation for user input.
- **Secrets:** Read from environment variables (`os.getenv()`). Never commit `.env`. Never bake secrets into the Docker image.
- **Structured output:** Gemini responses use `response_schema=DiagnosisResult` with `response_mime_type="application/json"` and `temperature=0.1`. Use `response.parsed`, not manual JSON parsing.
- **Config:** All GCP resource identifiers go in `config.py`. Import from there.
- **Logging:** `logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")` at module level. No print statements in production code.
- **Models:** Pydantic `BaseModel` for all API request/response shapes and for Gemini structured output schemas.
- **MCP server stdout:** `vehicle_mcp_server.py` redirects logging to stderr — keep it that way; the MCP protocol uses stdout.

---

## Running Locally

### Prerequisites

```bash
pip install -r requirements.txt
```

Create a `.env` file (never committed):
```
GEMINI_API_KEY=your_key_here
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
```

### FastAPI dev server

```bash
uvicorn api:app --reload --port 8080
```

### Docker Compose (API + Pub/Sub emulator)

```bash
docker-compose up --build
```

API available at `http://localhost:8080`. Health check: `GET /health`.

### Run the ETL pipeline

```bash
python etl.py
```

Reads `raw/*.csv` from `gs://vehicle-logs-sachin/`, loads to BigQuery, moves files to `processed/`.

### Run the ADK agent

```bash
python agent.py
```

### Run the MCP server + client

```bash
# Terminal 1
python vehicle_mcp_server.py

# Terminal 2
python mcp_client.py
```

### Run LangSmith evaluations

```bash
python run_evals.py
```

Requires `LANGCHAIN_API_KEY` in `.env`.
