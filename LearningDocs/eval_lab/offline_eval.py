"""
PHASE 1 -- PRE-PRODUCTION EVAL ("the gate")

Run this before every deploy: new prompt, new model version, new temperature,
new few-shot examples -- anything that changes what the LLM outputs.

Two kinds of checks, because LLM outputs have two kinds of fields:
  1. Hard labels (risk_level)      -> exact match against golden_dataset.py
  2. Free text (summary etc.)      -> LLM-as-judge score against a rubric

Both must clear a threshold or the run FAILS and (in a real pipeline) blocks
the deploy. This mirrors run_evals.py's LangSmith-based approach but adds the
quality-of-explanation dimension that exact-match alone misses.

Usage:
    python offline_eval.py
"""

import json
import statistics
from datetime import datetime, timezone

from golden_dataset import GOLDEN_SET
from model import run_model
from judge import judge_output

RISK_ACCURACY_THRESHOLD = 0.90   # safety-critical field: near-perfect required
JUDGE_SCORE_THRESHOLD = 3.5      # out of 5


def evaluate(drift: bool = False):
    rows = []
    for case in GOLDEN_SET:
        pred = run_model(
            case["vehicle_id"], case["error_code"], case["sensor"], case["value"],
            history=case["history"], drift=drift,
        )
        correct = int(pred["risk_level"] == case["expected_risk"])
        judge_score = judge_output(pred, case["rubric"])
        rows.append({
            "error_code": case["error_code"],
            "expected_risk": case["expected_risk"],
            "predicted_risk": pred["risk_level"],
            "risk_correct": correct,
            "judge_score": judge_score,
            "latency_ms": pred["latency_ms"],
        })

    accuracy = sum(r["risk_correct"] for r in rows) / len(rows)
    avg_judge = statistics.mean(r["judge_score"] for r in rows)
    p95_latency = sorted(r["latency_ms"] for r in rows)[int(len(rows) * 0.95) - 1]

    passed = accuracy >= RISK_ACCURACY_THRESHOLD and avg_judge >= JUDGE_SCORE_THRESHOLD

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_cases": len(rows),
        "risk_accuracy": round(accuracy, 3),
        "risk_accuracy_threshold": RISK_ACCURACY_THRESHOLD,
        "avg_judge_score": round(avg_judge, 2),
        "judge_score_threshold": JUDGE_SCORE_THRESHOLD,
        "p95_latency_ms": p95_latency,
        "gate_result": "PASS" if passed else "FAIL",
        "rows": rows,
    }
    return report


def print_report(report):
    print(f"\n{'='*60}")
    print("PRE-PRODUCTION EVAL -- offline_eval.py")
    print(f"{'='*60}")
    for r in report["rows"]:
        mark = "OK " if r["risk_correct"] else "XX "
        print(f"  [{mark}] {r['error_code']:6s} expected={r['expected_risk']:6s} "
              f"got={r['predicted_risk']:6s} judge={r['judge_score']}/5  "
              f"latency={r['latency_ms']}ms")

    print(f"\nRisk accuracy : {report['risk_accuracy']:.0%}  "
          f"(threshold {report['risk_accuracy_threshold']:.0%})")
    print(f"Avg judge score: {report['avg_judge_score']}/5  "
          f"(threshold {report['judge_score_threshold']}/5)")
    print(f"P95 latency   : {report['p95_latency_ms']}ms")
    print(f"\nGATE RESULT: {report['gate_result']}")
    if report["gate_result"] == "FAIL":
        print("  -> BLOCK deploy. Do not promote this model/prompt version.")
    else:
        print("  -> Safe to deploy. Proceed to production_monitor.py.")


if __name__ == "__main__":
    report = evaluate(drift=False)
    print_report(report)
    with open("offline_eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved offline_eval_report.json")
