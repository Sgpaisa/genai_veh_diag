from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from google.cloud import bigquery
from dotenv import load_dotenv
import os
from config import BQ_TABLE

load_dotenv()

# Same chain from langchain_diagnose.py
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1
)

bq = bigquery.Client()

def get_vehicle_history(vehicle_id: str) -> str:
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

Reply with ONLY one word on the first line: High, Medium, or Low.
Then explain briefly.""")

chain = prompt | llm


# Wrapper — adds history from BigQuery before invoking chain
def diagnose_for_eval(inputs: dict) -> dict:
    history = get_vehicle_history(inputs["vehicle_id"])
    result  = chain.invoke({**inputs, "history": history})
    # Extract first word as risk level
    first_word = result.content.strip().split()[0].replace("**","").strip()
    return {"risk_level": first_word}


# ── LangSmith dataset setup ──────────────────────────────

client  = Client()

# Create dataset only if it doesn't already exist
datasets = [d.name for d in client.list_datasets()]
if "vehicle-risk-eval-v1" not in datasets:
    dataset = client.create_dataset("vehicle-risk-eval-v1",
        description="OBD-II error code risk classification evaluation")
    client.create_examples(
        inputs=[
            {"error_code": "P0171", "vehicle_id": "VEH101", "sensor": "oxygen_sensor", "value": "0.12V"},
            {"error_code": "P0128", "vehicle_id": "VEH101", "sensor": "coolant_temp",  "value": "68C"},
            {"error_code": "P0300", "vehicle_id": "VEH102", "sensor": "misfire_count", "value": "18"},
            {"error_code": "P0420", "vehicle_id": "VEH103", "sensor": "catalyst",      "value": "0.71"},
            {"error_code": "P0171", "vehicle_id": "VEH102", "sensor": "oxygen_sensor", "value": "0.09V"},
        ],
        outputs=[
            {"expected_risk": "High"},
            {"expected_risk": "Medium"},
            {"expected_risk": "High"},
            {"expected_risk": "Medium"},
            {"expected_risk": "High"},
        ],
        dataset_id=dataset.id
    )
    print("Dataset created with 5 examples")
else:
    print("Dataset already exists — running eval on existing dataset")


# ── Evaluator ────────────────────────────────────────────

def risk_accuracy_evaluator(run, example):
    predicted = run.outputs.get("risk_level", "").strip()
    expected  = example.outputs.get("expected_risk", "").strip()
    correct   = int(predicted == expected)
    print(f"  {example.inputs.get('error_code')} → predicted={predicted} expected={expected} {'✓' if correct else '✗'}")
    return {"key": "risk_correct", "score": correct}


# ── Run evaluation ───────────────────────────────────────

print("\nRunning evaluation against 5 test cases...")
results = evaluate(
    diagnose_for_eval,
    data="vehicle-risk-eval-v1",
    evaluators=[risk_accuracy_evaluator],
    experiment_prefix="vehicle-diag-eval",
)

# Print summary
scores = [r["evaluation_results"]["results"][0].score
          for r in results._results if r.get("evaluation_results")]
if scores:
    accuracy = sum(scores) / len(scores)
    print(f"\nAccuracy: {accuracy:.0%}  ({sum(scores)}/{len(scores)} correct)")
    print("Check smith.langchain.com → Datasets → vehicle-risk-eval-v1 for full trace")
