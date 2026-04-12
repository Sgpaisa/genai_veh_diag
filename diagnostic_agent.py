from google import genai
from google.cloud import bigquery
from config import BQ_TABLE
import os

bq = bigquery.Client()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def get_fleet_summary():
    rows = bq.query(
        f"SELECT vehicle_id, COUNT(*) as error_count "
        f"FROM `{BQ_TABLE}` "
        f"GROUP BY vehicle_id ORDER BY error_count DESC"
    ).result()
    return [{"vehicle_id": r.vehicle_id, "error_count": r.error_count} for r in rows]

def get_vehicle_errors(vehicle_id):
    rows = bq.query(
        f"SELECT error_code, sensor, value FROM `{BQ_TABLE}` "
        f"WHERE vehicle_id = '{vehicle_id}' ORDER BY timestamp DESC LIMIT 20"
    ).result()
    return [{"code": r.error_code, "sensor": r.sensor, "value": r.value} for r in rows]

def ask(question):
    fleet = get_fleet_summary()
    top_vehicle = fleet[0]["vehicle_id"]
    errors = get_vehicle_errors(top_vehicle)

    prompt = f"""You are a vehicle diagnostic expert.

Fleet summary from BigQuery:
{fleet}

Detailed errors for most critical vehicle {top_vehicle}:
{errors}

Question: {question}

Give a clean professional diagnostic report with recommendations."""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    print("\n" + response.text)

ask("Give me full fleet health summary and which vehicle needs urgent attention?")
