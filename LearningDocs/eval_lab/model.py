"""
The "model under test".

Two backends:
  - REAL:  if GEMINI_API_KEY is set, calls Gemini exactly like diagnostics.py does
           (same DiagnosisResult schema, same temperature=0.1 convention).
  - MOCK:  deterministic rule-based stand-in, used so this whole lab runs with
           zero GCP/API setup. It also accepts a `drift` knob so we can SIMULATE
           a model/prompt regression on demand -- that's the thing production
           monitoring is supposed to catch.

Swap MOCK for REAL by setting GEMINI_API_KEY in .env. Nothing else changes --
that's the point: the eval harness shouldn't care which backend it's testing.
"""

import os
import random
import time
from dotenv import load_dotenv

load_dotenv()
USE_REAL = bool(os.getenv("GEMINI_API_KEY"))

# Rule-of-thumb "ground truth" the mock uses to answer correctly when NOT drifting.
_RISK_BY_CODE_PREFIX = {
    "P0171": "High", "P0300": "High", "P0217": "High", "P0011": "High",
    "P0128": "Medium", "P0420": "Medium", "P0562": "Medium",
    "P0442": "Low", "P0455": "Low",
}

_KEYWORDS_BY_CODE = {
    "P0171": ["lean mixture", "fuel/air ratio", "inspect the oxygen sensor and fuel system"],
    "P0300": ["misfire detected", "urgent inspection", "stop driving if severe"],
    "P0217": ["engine overheating", "stop the vehicle immediately", "check coolant system"],
    "P0011": ["camshaft timing", "variable valve timing", "schedule prompt inspection"],
    "P0128": ["thermostat", "engine running cooler than expected"],
    "P0420": ["catalytic converter efficiency below threshold", "not an immediate safety issue"],
    "P0562": ["low system voltage", "check battery and alternator"],
    "P0442": ["small evaporative emissions leak", "low urgency"],
    "P0455": ["large evaporative emissions leak", "emissions issue, not a drivability issue"],
}


def _mock_diagnose(vehicle_id, error_code, sensor, value, history="none", drift=False):
    """
    drift=False -> behaves like a healthy, well-tuned model (matches golden labels).
    drift=True  -> simulates a regressed model: vaguer text, occasional wrong risk
                   labels, higher latency. This is what "performance degradation
                   in production" looks like from the outside.
    """
    start = time.time()

    true_risk = _RISK_BY_CODE_PREFIX.get(error_code, "Medium")
    keywords = _KEYWORDS_BY_CODE.get(error_code, ["general diagnostic issue"])

    if not drift:
        risk = true_risk
        summary = f"OBD-II {error_code} on {sensor} ({value}). " + "; ".join(keywords) + "."
        root_causes = keywords[:2]
        actions = [keywords[-1]]
        latency_ms = random.randint(180, 320)
    else:
        # Degraded behavior: ~40% chance of wrong/understated risk label,
        # generic text that misses the rubric keywords, slower responses.
        if random.random() < 0.4:
            order = ["Low", "Medium", "High"]
            idx = max(0, order.index(true_risk) - 1)  # understate severity
            risk = order[idx]
        else:
            risk = true_risk
        summary = f"Error code {error_code} detected. Recommend general inspection."
        root_causes = ["unspecified sensor anomaly"]
        actions = ["schedule a routine service visit"]
        latency_ms = random.randint(600, 1400)

    elapsed = time.time() - start
    return {
        "risk_level": risk,
        "root_causes": root_causes,
        "immediate_actions": actions,
        "summary": summary,
        "latency_ms": latency_ms,
        "_backend": "mock",
    }


def _real_diagnose(vehicle_id, error_code, sensor, value, history="none", drift=False):
    # Real Gemini call, same shape as diagnostics.py. `drift` isn't meaningful
    # here (you can't inject drift into a real model on demand) -- it's only
    # honored by the mock backend for demo purposes.
    from google import genai
    from google.genai import types
    from pydantic import BaseModel
    from typing import Literal
    from config import GEMINI_MODEL

    class DiagnosisResult(BaseModel):
        risk_level: Literal["High", "Medium", "Low"]
        root_causes: list[str]
        immediate_actions: list[str]
        summary: str

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    prompt = f"""You are an expert automotive diagnostic engineer.
Vehicle: {vehicle_id}
OBD-II Error Code: {error_code}
Sensor: {sensor}
Reading: {value}
Recent error history: {history}
Provide a structured diagnosis."""

    start = time.time()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_schema=DiagnosisResult,
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )
    latency_ms = int((time.time() - start) * 1000)
    result = response.parsed
    return {
        "risk_level": result.risk_level,
        "root_causes": result.root_causes,
        "immediate_actions": result.immediate_actions,
        "summary": result.summary,
        "latency_ms": latency_ms,
        "_backend": "gemini",
    }


def run_model(vehicle_id, error_code, sensor, value, history="none", drift=False):
    if USE_REAL:
        return _real_diagnose(vehicle_id, error_code, sensor, value, history, drift)
    return _mock_diagnose(vehicle_id, error_code, sensor, value, history, drift)
