from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from google.cloud import bigquery, aiplatform
from diagnostics import diagnose, DiagnosisResult
from dotenv import load_dotenv
import logging
from config import PROJECT_ID, REGION, BQ_TABLE, VS_ENDPOINT_ID

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
aiplatform.init(project=PROJECT_ID, location=REGION)

app = FastAPI(title="Vehicle Diagnostic API", version="2.0",
              description="Fleet OBD-II diagnostics — Gemini + Vertex AI + Google ADK")
bq = bigquery.Client()


class ErrorRecord(BaseModel):
    error_code: str | None
    sensor:     str | None
    value:      float | None
    timestamp:  str

class VehicleResponse(BaseModel):
    vehicle_id:   str
    total_errors: int
    errors:       list[ErrorRecord]

class FleetEntry(BaseModel):
    error_code:        str
    occurrences:       int
    vehicles_affected: int

class FleetResponse(BaseModel):
    fleet_top_errors: list[FleetEntry]

class DiagnoseResponse(BaseModel):
    vehicle_id:        str
    error_code:        str
    risk_level:        str
    root_causes:       list[str]
    immediate_actions: list[str]
    summary:           str


@app.get("/health")
async def health():
    return {"status": "ok", "service": "Vehicle Diagnostic API", "version": "2.0"}


@app.get("/vehicle/{vehicle_id}", response_model=VehicleResponse)
async def vehicle_errors(vehicle_id: str):
    rows = bq.query(
        f"SELECT error_code, sensor, value, timestamp FROM `{BQ_TABLE}` "
        f"WHERE vehicle_id = @v ORDER BY timestamp DESC LIMIT 20",
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("v", "STRING", vehicle_id)])
    ).result()
    errors = [ErrorRecord(error_code=r.error_code, sensor=r.sensor,
                          value=r.value, timestamp=str(r.timestamp)) for r in rows]
    if not errors:
        raise HTTPException(status_code=404, detail=f"No data found for {vehicle_id}")
    return VehicleResponse(vehicle_id=vehicle_id, total_errors=len(errors), errors=errors)


@app.get("/fleet/summary", response_model=FleetResponse)
async def fleet_summary():
    rows = bq.query(
        f"SELECT error_code, COUNT(*) AS occ, APPROX_COUNT_DISTINCT(vehicle_id) AS vehicles "
        f"FROM `{BQ_TABLE}` WHERE error_code IS NOT NULL "
        f"GROUP BY error_code ORDER BY occ DESC LIMIT 10"
    ).result()
    return FleetResponse(fleet_top_errors=[
        FleetEntry(error_code=r.error_code, occurrences=r.occ, vehicles_affected=r.vehicles)
        for r in rows])


@app.get("/diagnose/{vehicle_id}/{error_code}", response_model=DiagnoseResponse)
async def diagnose_route(vehicle_id: str, error_code: str,
                          sensor: str = Query(default="unknown"),
                          value:  str = Query(default="unknown")):
    result = diagnose(vehicle_id, error_code, sensor, value)
    return DiagnoseResponse(
        vehicle_id=vehicle_id, error_code=error_code,
        risk_level=result.risk_level, root_causes=result.root_causes,
        immediate_actions=result.immediate_actions, summary=result.summary)
