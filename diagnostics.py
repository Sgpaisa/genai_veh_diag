from google import genai
from google.genai import types
from google.cloud import bigquery
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv
import logging, os
from config import PROJECT_ID, BQ_TABLE, GEMINI_MODEL

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class DiagnosisResult(BaseModel):
    risk_level:        Literal["High", "Medium", "Low"]
    root_causes:       list[str]
    immediate_actions: list[str]
    summary:           str
	
def get_vehicle_history(vehicle_id: str) -> list[str]:
    bq   = bigquery.Client()
    rows = bq.query(
        f"SELECT error_code FROM `{BQ_TABLE}` "
        f"WHERE vehicle_id = @v AND error_code IS NOT NULL "
        f"ORDER BY timestamp DESC LIMIT 5",
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("v", "STRING", vehicle_id)]
        )
    ).result()
    return [r.error_code for r in rows]


def diagnose(vehicle_id: str, error_code: str, sensor: str, value: str) -> DiagnosisResult:
    history      = get_vehicle_history(vehicle_id)
    history_text = ", ".join(history) if history else "none"
    prompt = f"""You are an expert automotive diagnostic engineer.
Vehicle: {vehicle_id}
OBD-II Error Code: {error_code}
Sensor: {sensor}
Reading: {value}
Recent error history: {history_text}
Provide a structured diagnosis."""
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_schema=DiagnosisResult,
            response_mime_type="application/json",
            temperature=0.1,
        )
    )
    result = response.parsed
    logging.info(f"Diagnosed {vehicle_id}/{error_code} → Risk={result.risk_level}")
    return result


if __name__ == "__main__":
    result = diagnose("VEH101", "P0171", "oxygen_sensor", "0.12V")
    print(f"Risk Level : {result.risk_level}")
    print(f"Root Causes: {result.root_causes}")
    print(f"Actions    : {result.immediate_actions}")
    print(f"Summary    : {result.summary}")
