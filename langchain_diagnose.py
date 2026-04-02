from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from google.cloud import bigquery
from dotenv import load_dotenv
import os
from config import BQ_TABLE

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1
)

bq = bigquery.Client()


def get_vehicle_history(vehicle_id: str) -> str:
    """Fetch last 5 error codes from BigQuery for context."""
    rows = bq.query(
        f"SELECT error_code FROM `{BQ_TABLE}` "
        f"WHERE vehicle_id = @v AND error_code IS NOT NULL "
        f"ORDER BY timestamp DESC LIMIT 5",
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("v", "STRING", vehicle_id)]
        )
    ).result()
    codes = [r.error_code for r in rows]
    return ", ".join(codes) if codes else "none"


prompt = PromptTemplate.from_template("""You are an expert automotive diagnostic engineer.
Vehicle: {vehicle_id}
OBD-II Error Code: {error_code}
Sensor: {sensor}
Reading: {value}
Recent error history: {history}

Classify risk as High / Medium / Low.
List 3 root causes.
List 3 immediate actions.
Give a one-sentence summary.""")

chain = prompt | llm


if __name__ == "__main__":
    test_cases = [
        {"vehicle_id": "VEH101", "error_code": "P0171", "sensor": "oxygen_sensor", "value": "0.12V"},
        {"vehicle_id": "VEH102", "error_code": "P0300", "sensor": "misfire_count", "value": "18"},
        {"vehicle_id": "VEH103", "error_code": "P0128", "sensor": "coolant_temp",  "value": "68C"},
    ]
    for t in test_cases:
        history = get_vehicle_history(t["vehicle_id"])
        print(f"\nDiagnosing {t['vehicle_id']} / {t['error_code']} (history: {history})")
        result = chain.invoke({**t, "history": history})
        print(result.content[:300])
        print("--- trace sent to LangSmith ---")
