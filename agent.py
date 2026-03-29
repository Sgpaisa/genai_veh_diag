from google.genai.types import Content, Part
import asyncio, logging
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.cloud import bigquery
from etl import read_raw_files, clean, load_to_bigquery, embed_and_index
from diagnostics import diagnose
import pandas as pd
from config import BQ_TABLE


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def fetch_and_load() -> dict:
    """Fetch raw OBD-II CSVs from Cloud Storage, clean, load to BigQuery, index embeddings."""
    blobs, bucket = read_raw_files()
    if not blobs:
        return {"status": "no_files", "message": "No CSV in raw/. Upload a file first."}
    frames = []
    for blob in blobs:
        df = clean(blob.download_as_text(), blob.name)
        bucket.blob(blob.name.replace("raw/","processed/")).upload_from_string(
            df.to_csv(index=False), content_type="text/csv")
        blob.delete()
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    load_to_bigquery(combined)
    embed_and_index(combined)
    return {
        "status":   "success",
        "rows":     len(combined),
        "vehicles": combined["vehicle_id"].unique().tolist(),
        "errors":   combined.dropna(subset=["error_code"])[["vehicle_id","error_code","sensor","value"]].to_dict("records"),
    }
	
def diagnose_error(vehicle_id: str, error_code: str, sensor: str, value: str) -> dict:
    """Run Gemini AI diagnosis for one OBD-II error. Returns risk_level, root_causes, actions."""
    result = diagnose(vehicle_id, error_code, sensor, value)
    return {
        "vehicle_id":        vehicle_id,
        "error_code":        error_code,
        "risk_level":        result.risk_level,
        "root_causes":       result.root_causes,
        "immediate_actions": result.immediate_actions,
        "summary":           result.summary,
    }
	
def send_alert(vehicle_id: str, error_code: str, summary: str) -> dict:
    """Send URGENT maintenance alert. Call ONLY when risk_level is High."""
    alert = {"alert_id": f"ALERT-{vehicle_id}-{error_code}", "priority": "URGENT",
              "vehicle_id": vehicle_id, "error_code": error_code,
              "action": "Inspect within 4 hours", "summary": summary}
    logging.info(f"ALERT SENT: {alert}")
    return {"sent": True, "alert": alert}


def fleet_summary() -> dict:
    """Query BigQuery for vehicle error summary. Call after fetch_and_load."""
    bq   = bigquery.Client()
    rows = bq.query(
        f"SELECT vehicle_id, COUNT(*) as cnt, STRING_AGG(DISTINCT error_code, ', ') as codes "
        f"FROM `{BQ_TABLE}` WHERE error_code IS NOT NULL "
        f"GROUP BY vehicle_id ORDER BY cnt DESC LIMIT 10"
    ).result()
    return {"vehicles": [{"vehicle_id": r.vehicle_id, "error_count": r.cnt, "error_codes": r.codes} for r in rows]}

agent = LlmAgent(
    name        = "VehicleFleetDiagnosticAgent",
    model       = "gemini-2.5-flash",
    description = "Autonomous OBD-II fleet diagnostic agent",
    instruction = """You are an autonomous vehicle fleet diagnostic agent.
Workflow:
1. Call fetch_and_load — always first
2. Call fleet_summary — see what vehicles/errors exist
3. For each vehicle + error: call diagnose_error
4. If risk_level is "High" → call send_alert immediately
5. If "Medium" or "Low" → log only, no alert
6. End with a complete fleet status report""",
    tools=[FunctionTool(fetch_and_load), FunctionTool(diagnose_error),
           FunctionTool(send_alert), FunctionTool(fleet_summary)]
)


async def run() -> None:
    sessions = InMemorySessionService()
    runner   = Runner(agent=agent, app_name="vehicle_diagnostics", session_service=sessions)
    session  = await sessions.create_session(app_name="vehicle_diagnostics", user_id="fleet_ops")
    message = Content(role="user", parts=[Part(text="Process all new vehicle log files. Diagnose every error. Send alerts where needed. Give me a complete fleet status report.")])
    logging.info("========== ADK AGENT STARTED ==========")
    async for event in runner.run_async(user_id="fleet_ops", session_id=session.id, new_message=message):
        if event.is_final_response() and event.content:
            print("\n" + event.content.parts[0].text)


if __name__ == "__main__":
    asyncio.run(run())
