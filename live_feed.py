"""
Live Fleet Feed — Full Pipeline
=================================
GET /fleet/live-feed  → JSON with full pipeline output (BigQuery → RAG → Gemini)
GET /fleet/dashboard  → Live HTML dashboard, auto-refreshes every 30 seconds

Full flow per request:
  1. BigQuery  — fetch all errors from last 2 hours across all garages
  2. Vector Search (RAG) — retrieve similar historical errors for each vehicle
  3. Gemini (LLM) — diagnose each vehicle grounded in RAG context
  4. Aggregate — group by garage, compute risk summary
  5. Return JSON + serve HTML dashboard
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from google.cloud import bigquery
from google import genai
from google.genai import types
from dotenv import load_dotenv
from config import PROJECT_ID, BQ_TABLE, GEMINI_MODEL, EMBEDDING_MODEL, VS_ENDPOINT_ID
from diagnostics import diagnose, DiagnosisResult

load_dotenv()
logger = logging.getLogger(__name__)
router = APIRouter()

bq     = bigquery.Client()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ── Pydantic response models ──────────────────────────────────────────────────

class VehicleDiagnosis(BaseModel):
    vehicle_id:        str
    error_code:        str
    sensor:            str
    value:             float | None
    risk_level:        str
    root_causes:       list[str]
    immediate_actions: list[str]
    summary:           str
    similar_past_errors: list[str]   # RAG context used


class GarageFeed(BaseModel):
    garage_id:      str
    location:       str
    total_errors:   int
    high_risk:      int
    medium_risk:    int
    low_risk:       int
    vehicles:       list[VehicleDiagnosis]
    last_updated:   str


class LiveFeedResponse(BaseModel):
    generated_at:    str
    total_garages:   int
    total_vehicles:  int
    critical_alerts: int
    pipeline_stages: dict          # shows which pipeline stages ran
    garages:         list[GarageFeed]


# ── Step 1: BigQuery — fetch recent errors across all garages ─────────────────

def fetch_recent_errors(hours: int = 2) -> list[dict]:
    """
    Pull all errors from the last N hours across every garage.
    garage_id is extracted from source_file path: raw/garage_mumbai_01/fleet.csv
    """
    query = f"""
        SELECT
            vehicle_id,
            error_code,
            sensor,
            value,
            timestamp,
            source_file,
            -- Extract garage_id from path: raw/garage_mumbai_01/fleet.csv → garage_mumbai_01
            COALESCE(
                REGEXP_EXTRACT(source_file, r'raw/([^/]+)/'),
                'garage_default'
            ) AS garage_id
        FROM `{BQ_TABLE}`
        WHERE
            error_code IS NOT NULL
            AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @hours HOUR)
        ORDER BY timestamp DESC
        LIMIT 200
    """
    job = bq.query(query, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("hours", "INT64", hours)]
    ))
    rows = list(job.result())
    logger.info(f"BigQuery: fetched {len(rows)} errors from last {hours}h")
    return [dict(r) for r in rows]


# ── Step 2: RAG — Vector Search for similar historical errors ─────────────────

def fetch_similar_errors(vehicle_id: str, error_code: str, sensor: str) -> list[str]:
    """
    Embed the symptom description → query Vertex AI Vector Search
    → return top-3 similar historical error descriptions.
    This is the RAG retrieval step — grounds Gemini in real fleet history.
    """
    if not VS_ENDPOINT_ID:
        return []

    try:
        from vertexai.language_models import TextEmbeddingModel
        from google.cloud import aiplatform

        description = (
            f"Vehicle {vehicle_id} reported OBD-II fault {error_code} "
            f"on {sensor} sensor."
        )
        model     = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
        query_vec = model.get_embeddings([description])[0].values

        endpoint  = aiplatform.MatchingEngineIndexEndpoint(VS_ENDPOINT_ID)
        response  = endpoint.find_neighbors(
            deployed_index_id="vehicle_errors_v3",
            queries=[query_vec],
            num_neighbors=3
        )
        similar = [
            f"{n.id} (similarity: {round(n.distance * 100)}%)"
            for n in response[0]
        ]
        logger.info(f"RAG: found {len(similar)} similar errors for {vehicle_id}/{error_code}")
        return similar

    except Exception as e:
        logger.warning(f"RAG retrieval skipped: {e}")
        return []


# ── Step 3: Gemini — LLM diagnosis grounded in RAG context ───────────────────

def diagnose_with_rag_context(
    vehicle_id: str,
    error_code: str,
    sensor: str,
    value: str,
    similar_errors: list[str]
) -> DiagnosisResult:
    """
    Diagnose using Gemini with RAG context injected into the prompt.
    Similar past errors retrieved from Vector Search are included
    so Gemini grounds its diagnosis in real fleet history — not just training data.
    """
    rag_context = (
        "\n".join(f"  - {s}" for s in similar_errors)
        if similar_errors
        else "  None available"
    )

    prompt = f"""You are an expert automotive diagnostic engineer for a multi-location fleet.

Vehicle ID   : {vehicle_id}
Error Code   : {error_code}
Sensor       : {sensor}
Reading      : {value}

Similar errors from fleet history (retrieved from vector search):
{rag_context}

Using the fleet history above as context, provide a structured diagnosis.
Do not repeat the similar error IDs in your summary."""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_schema=DiagnosisResult,
            response_mime_type="application/json",
            temperature=0.1,
            system_instruction=(
                "You are an expert automotive diagnostic engineer. "
                "Always respond with structured JSON. "
                "Never fabricate error codes or vehicle IDs not given to you."
            )
        )
    )
    return response.parsed


# ── Step 4: Aggregate by garage ───────────────────────────────────────────────

GARAGE_LOCATIONS = {
    "garage_mumbai_01": "Mumbai — Andheri",
    "garage_mumbai_02": "Mumbai — Bandra",
    "garage_delhi_01":  "Delhi — Connaught Place",
    "garage_delhi_02":  "Delhi — Dwarka",
    "garage_blore_01":  "Bangalore — Whitefield",
    "garage_blore_02":  "Bangalore — Electronic City",
    "garage_default":   "Unknown Location",
}


def build_garage_feed(garage_id: str, diagnoses: list[VehicleDiagnosis]) -> GarageFeed:
    return GarageFeed(
        garage_id    = garage_id,
        location     = GARAGE_LOCATIONS.get(garage_id, garage_id.replace("_", " ").title()),
        total_errors = len(diagnoses),
        high_risk    = sum(1 for d in diagnoses if d.risk_level == "High"),
        medium_risk  = sum(1 for d in diagnoses if d.risk_level == "Medium"),
        low_risk     = sum(1 for d in diagnoses if d.risk_level == "Low"),
        vehicles     = diagnoses,
        last_updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )


# ── Main pipeline endpoint ────────────────────────────────────────────────────

@router.get("/fleet/live-feed", response_model=LiveFeedResponse)
async def live_feed(hours: int = 2):
    """
    Full pipeline: BigQuery → RAG (Vector Search) → Gemini LLM → JSON response.
    Called every 30 seconds by the dashboard. Shows live status across all garages.
    """
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    pipeline_stages = {
        "1_bigquery":      "pending",
        "2_rag_retrieval": "pending",
        "3_llm_diagnosis": "pending",
        "4_aggregation":   "pending",
    }

    # ── Stage 1: BigQuery ────────────────────────────────────────────────────
    try:
        raw_errors = fetch_recent_errors(hours=hours)
        pipeline_stages["1_bigquery"] = f"ok — {len(raw_errors)} errors fetched"
    except Exception as e:
        pipeline_stages["1_bigquery"] = f"error: {e}"
        raw_errors = []

    if not raw_errors:
        return LiveFeedResponse(
            generated_at    = generated_at,
            total_garages   = 0,
            total_vehicles  = 0,
            critical_alerts = 0,
            pipeline_stages = pipeline_stages,
            garages         = [],
        )

    # Group errors by garage
    from collections import defaultdict
    by_garage: dict[str, list[dict]] = defaultdict(list)
    for row in raw_errors:
        by_garage[row["garage_id"]].append(row)

    # ── Stages 2 + 3: RAG → LLM per vehicle ─────────────────────────────────
    garage_feeds: list[GarageFeed] = []
    total_high = 0

    for garage_id, errors in by_garage.items():
        diagnoses: list[VehicleDiagnosis] = []

        for err in errors:
            vehicle_id = err["vehicle_id"]
            error_code = err["error_code"]
            sensor     = err.get("sensor", "unknown")
            value      = err.get("value")
            value_str  = str(value) if value is not None else "unknown"

            # Stage 2: RAG — get similar past errors from Vector Search
            try:
                similar = fetch_similar_errors(vehicle_id, error_code, sensor)
                pipeline_stages["2_rag_retrieval"] = "ok"
            except Exception as e:
                similar = []
                pipeline_stages["2_rag_retrieval"] = f"partial: {e}"

            # Stage 3: LLM — Gemini diagnosis grounded in RAG context
            try:
                result = diagnose_with_rag_context(
                    vehicle_id, error_code, sensor, value_str, similar
                )
                pipeline_stages["3_llm_diagnosis"] = "ok"
            except Exception as e:
                logger.error(f"Gemini failed for {vehicle_id}/{error_code}: {e}")
                pipeline_stages["3_llm_diagnosis"] = f"partial: {e}"
                continue

            diagnoses.append(VehicleDiagnosis(
                vehicle_id          = vehicle_id,
                error_code          = error_code,
                sensor              = sensor,
                value               = value,
                risk_level          = result.risk_level,
                root_causes         = result.root_causes,
                immediate_actions   = result.immediate_actions,
                summary             = result.summary,
                similar_past_errors = similar,
            ))

        # Stage 4: Aggregate into garage feed
        feed = build_garage_feed(garage_id, diagnoses)
        total_high += feed.high_risk
        garage_feeds.append(feed)

    pipeline_stages["4_aggregation"] = (
        f"ok — {len(garage_feeds)} garages, {total_high} critical alerts"
    )

    # Sort: garages with High risk first
    garage_feeds.sort(key=lambda g: g.high_risk, reverse=True)

    return LiveFeedResponse(
        generated_at    = generated_at,
        total_garages   = len(garage_feeds),
        total_vehicles  = sum(len(g.vehicles) for g in garage_feeds),
        critical_alerts = total_high,
        pipeline_stages = pipeline_stages,
        garages         = garage_feeds,
    )


# ── Live HTML Dashboard ───────────────────────────────────────────────────────

@router.get("/fleet/dashboard", response_class=HTMLResponse)
async def fleet_dashboard():
    """
    Live HTML dashboard served at /fleet/dashboard.
    Auto-refreshes every 30 seconds by calling /fleet/live-feed in the background.
    No page reload — JavaScript fetches fresh data and updates the DOM.
    """
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fleet Live Feed</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; font-size: 13px; }
  .header { background: #1e293b; padding: 14px 20px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #334155; position: sticky; top: 0; z-index: 10; }
  .header h1 { font-size: 16px; font-weight: 700; color: #f1f5f9; }
  .header .meta { font-size: 11px; color: #64748b; }
  .stats { display: flex; gap: 12px; padding: 14px 20px; background: #0f172a; }
  .stat { background: #1e293b; border-radius: 8px; padding: 12px 18px; flex: 1; border: 1px solid #334155; }
  .stat .label { font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: .5px; }
  .stat .value { font-size: 22px; font-weight: 700; margin-top: 4px; }
  .stat.red .value { color: #f87171; }
  .stat.yellow .value { color: #fbbf24; }
  .stat.green .value { color: #34d399; }
  .stat.blue .value { color: #60a5fa; }
  .pipeline { margin: 0 20px 14px; background: #1e293b; border-radius: 8px; padding: 10px 14px; border: 1px solid #334155; }
  .pipeline h3 { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: .5px; margin-bottom: 8px; }
  .pipe-stages { display: flex; gap: 8px; flex-wrap: wrap; }
  .pipe-stage { font-size: 10px; padding: 3px 10px; border-radius: 20px; background: #0f172a; border: 1px solid #334155; color: #94a3b8; }
  .pipe-stage.ok { border-color: #34d399; color: #34d399; }
  .pipe-stage.err { border-color: #f87171; color: #f87171; }
  .garages { padding: 0 20px 20px; display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 14px; }
  .garage { background: #1e293b; border-radius: 10px; border: 1px solid #334155; overflow: hidden; }
  .garage.has-high { border-color: #f87171; }
  .garage-header { padding: 12px 14px; background: #0f172a; display: flex; justify-content: space-between; align-items: center; }
  .garage-name { font-weight: 700; font-size: 13px; }
  .garage-loc { font-size: 10px; color: #64748b; margin-top: 2px; }
  .risk-pills { display: flex; gap: 5px; }
  .pill { font-size: 10px; font-weight: 700; padding: 3px 9px; border-radius: 20px; }
  .pill.high { background: #7f1d1d; color: #fca5a5; }
  .pill.med { background: #78350f; color: #fcd34d; }
  .pill.low { background: #14532d; color: #86efac; }
  .vehicles { padding: 8px; }
  .vehicle { background: #0f172a; border-radius: 7px; padding: 10px 12px; margin-bottom: 6px; border-left: 3px solid #334155; }
  .vehicle.high { border-left-color: #f87171; }
  .vehicle.medium { border-left-color: #fbbf24; }
  .vehicle.low { border-left-color: #34d399; }
  .v-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
  .v-id { font-weight: 700; font-size: 12px; }
  .v-code { font-size: 10px; background: #1e293b; padding: 2px 8px; border-radius: 4px; color: #94a3b8; }
  .v-summary { font-size: 11px; color: #94a3b8; line-height: 1.5; margin-bottom: 5px; }
  .v-actions { font-size: 10px; color: #34d399; }
  .v-rag { font-size: 10px; color: #60a5fa; margin-top: 4px; }
  .badge-risk { font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 4px; }
  .badge-risk.High { background: #7f1d1d; color: #fca5a5; }
  .badge-risk.Medium { background: #78350f; color: #fcd34d; }
  .badge-risk.Low { background: #14532d; color: #86efac; }
  .refresh-bar { height: 3px; background: #334155; position: fixed; bottom: 0; left: 0; width: 100%; }
  .refresh-progress { height: 100%; background: #3b82f6; width: 100%; transition: width 30s linear; }
  .empty { padding: 40px; text-align: center; color: #475569; }
  #status { font-size: 11px; color: #64748b; }
  #status.loading { color: #fbbf24; }
  #status.error { color: #f87171; }
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>🚗 Fleet Live Feed — All Locations</h1>
    <div class="meta">Full pipeline: BigQuery → RAG (Vector Search) → Gemini LLM → Live Dashboard</div>
  </div>
  <div id="status">Loading...</div>
</div>

<div class="stats">
  <div class="stat blue">
    <div class="label">Garages Online</div>
    <div class="value" id="stat-garages">—</div>
  </div>
  <div class="stat blue">
    <div class="label">Vehicles (2h)</div>
    <div class="value" id="stat-vehicles">—</div>
  </div>
  <div class="stat red">
    <div class="label">Critical Alerts</div>
    <div class="value" id="stat-critical">—</div>
  </div>
  <div class="stat green">
    <div class="label">Last Updated</div>
    <div class="value" style="font-size:13px;margin-top:6px" id="stat-time">—</div>
  </div>
</div>

<div class="pipeline">
  <h3>Pipeline Status</h3>
  <div class="pipe-stages" id="pipeline-stages">—</div>
</div>

<div class="garages" id="garages-container">
  <div class="empty">Fetching live data from all garage locations...</div>
</div>

<div class="refresh-bar"><div class="refresh-progress" id="progress-bar"></div></div>

<script>
async function fetchFeed() {
  const status = document.getElementById('status');
  status.className = 'loading';
  status.textContent = 'Fetching pipeline data...';

  try {
    const res  = await fetch('/fleet/live-feed?hours=2');
    const data = await res.json();

    // Stats
    document.getElementById('stat-garages').textContent  = data.total_garages;
    document.getElementById('stat-vehicles').textContent = data.total_vehicles;
    document.getElementById('stat-critical').textContent = data.critical_alerts;
    document.getElementById('stat-time').textContent     = data.generated_at.split(' ')[1] + ' UTC';

    // Pipeline stages
    const stagesEl = document.getElementById('pipeline-stages');
    stagesEl.innerHTML = Object.entries(data.pipeline_stages).map(([k, v]) => {
      const isOk  = v.startsWith('ok');
      const isErr = v.startsWith('error');
      const cls   = isOk ? 'ok' : isErr ? 'err' : '';
      const icon  = isOk ? '✓' : isErr ? '✗' : '…';
      const label = k.replace(/_/g,' ').replace(/^\\d /,'');
      return `<span class="pipe-stage ${cls}">${icon} ${label}: ${v}</span>`;
    }).join('');

    // Garages
    const container = document.getElementById('garages-container');
    if (!data.garages.length) {
      container.innerHTML = '<div class="empty">No errors reported in the last 2 hours across any garage.</div>';
    } else {
      container.innerHTML = data.garages.map(g => `
        <div class="garage ${g.high_risk > 0 ? 'has-high' : ''}">
          <div class="garage-header">
            <div>
              <div class="garage-name">📍 ${g.location}</div>
              <div class="garage-loc">${g.garage_id} · ${g.total_errors} errors · ${g.last_updated}</div>
            </div>
            <div class="risk-pills">
              ${g.high_risk   ? `<span class="pill high">🔴 ${g.high_risk} High</span>` : ''}
              ${g.medium_risk ? `<span class="pill med">🟡 ${g.medium_risk} Med</span>` : ''}
              ${g.low_risk    ? `<span class="pill low">🟢 ${g.low_risk} Low</span>` : ''}
            </div>
          </div>
          <div class="vehicles">
            ${g.vehicles.map(v => `
              <div class="vehicle ${v.risk_level.toLowerCase()}">
                <div class="v-header">
                  <span class="v-id">${v.vehicle_id}</span>
                  <div style="display:flex;gap:6px;align-items:center">
                    <span class="v-code">${v.error_code} · ${v.sensor}</span>
                    <span class="badge-risk ${v.risk_level}">${v.risk_level}</span>
                  </div>
                </div>
                <div class="v-summary">${v.summary}</div>
                <div class="v-actions">⚡ ${v.immediate_actions.slice(0,2).join(' · ')}</div>
                ${v.similar_past_errors.length
                  ? `<div class="v-rag">🔍 RAG context: ${v.similar_past_errors[0]}</div>`
                  : ''}
              </div>
            `).join('')}
          </div>
        </div>
      `).join('');
    }

    status.className = '';
    status.textContent = `Live · refreshes in 30s`;

  } catch (err) {
    status.className = 'error';
    status.textContent = `Error: ${err.message}`;
  }

  // Reset progress bar animation for next 30s cycle
  const bar = document.getElementById('progress-bar');
  bar.style.transition = 'none';
  bar.style.width = '100%';
  setTimeout(() => {
    bar.style.transition = 'width 30s linear';
    bar.style.width = '0%';
  }, 100);
}

// Fetch immediately, then every 30 seconds
fetchFeed();
setInterval(fetchFeed, 30000);
</script>
</body>
</html>"""
    return HTMLResponse(content=html)
