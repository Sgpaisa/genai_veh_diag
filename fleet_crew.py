from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from google.cloud import bigquery
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "")

from config import BQ_TABLE


# ── Real BigQuery tool — agent calls this to get actual data ─────────
@tool("Query BigQuery fleet data")
def query_bigquery_fleet(query: str) -> str:
    """Query the BigQuery vehicle_diagnostics.error_logs table.
    Use this tool to get real vehicle error data from the fleet."""
    bq   = bigquery.Client()
    rows = bq.query(
        f"SELECT vehicle_id, error_code, COUNT(*) as cnt "
        f"FROM `{BQ_TABLE}` "
        f"WHERE error_code IS NOT NULL "
        f"GROUP BY vehicle_id, error_code "
        f"ORDER BY vehicle_id, cnt DESC"
    ).result()
    output = ""
    for r in rows:
        output += f"Vehicle {r.vehicle_id}: {r.error_code} ({r.cnt} times)\n"
    return output if output else "No data found in BigQuery"


# ── Agents ────────────────────────────────────────────────────────────
data_analyst = Agent(
    role="Fleet Data Analyst",
    goal="Analyse BigQuery vehicle error data and identify patterns",
    backstory="Expert automotive data analyst with 10 years fleet management experience",
    llm="gemini/gemini-2.5-flash",
    tools=[query_bigquery_fleet],   # ← real BigQuery tool attached
    verbose=True
)

diagnostic_expert = Agent(
    role="Senior Automotive Diagnostic Engineer",
    goal="Diagnose OBD-II error codes and assess risk levels",
    backstory="Expert diagnostician who has diagnosed 10,000+ vehicle faults",
    llm="gemini/gemini-2.5-flash",
    verbose=True
)

report_writer = Agent(
    role="Fleet Report Writer",
    goal="Produce a clear, actionable executive fleet health report",
    backstory="Technical writer specialising in automotive fleet management reports",
    llm="gemini/gemini-2.5-flash",
    verbose=True
)


# ── Tasks ─────────────────────────────────────────────────────────────
analyse_task = Task(
    description="""Use the query_bigquery_fleet tool to fetch real vehicle error data
    from BigQuery table vehicle_diagnostics.error_logs in project vehicle-diagnostics-491610.
    Summarise: how many vehicles, what error codes, how many occurrences each.""",
    expected_output="Bullet list of real vehicles and their error codes with counts from BigQuery",
    agent=data_analyst
)

diagnose_task = Task(
    description="""Using the REAL data from the analyst, diagnose each OBD-II error code.
    For P0171: lean fuel mixture, High risk. For P0300: random misfire, High risk.
    For P0128: thermostat fault, Medium risk. For P0420: catalyst efficiency, Medium risk.
    List risk level and key action for each error code found.""",
    expected_output="Per-error diagnosis with risk level and recommended action",
    agent=diagnostic_expert,
    context=[analyse_task]
)

report_task = Task(
    description="""Write a one-page executive fleet health report based on:
    1. The analyst's REAL vehicle data summary from BigQuery
    2. The diagnostic expert's risk assessment
    Format: Executive Summary, High-Risk Vehicles (immediate action), Monitor List.""",
    expected_output="Professional fleet health report in markdown format",
    agent=report_writer,
    context=[analyse_task, diagnose_task]
)


# ── Crew ──────────────────────────────────────────────────────────────
crew = Crew(
    agents=[data_analyst, diagnostic_expert, report_writer],
    tasks=[analyse_task, diagnose_task, report_task],
    process=Process.sequential,
    verbose=True
)


if __name__ == "__main__":
    result = crew.kickoff()
    print("\n" + "="*60)
    print("FLEET HEALTH REPORT — REAL DATA FROM BIGQUERY")
    print("="*60)
    print(result)
