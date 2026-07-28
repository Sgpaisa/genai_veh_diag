"""
PHASE 2 -- PRODUCTION MONITORING (post-deploy eval, continuous)

Offline eval (phase 1) only proves the model was good on 10 golden cases at
one point in time. Production traffic is different: new error codes, edge
cases, and the model/prompt can silently regress (provider updates the model
behind the scenes, someone tweaks the prompt without re-running the gate,
context distribution shifts, etc). You only catch that by watching live.

What this simulates:
  - a stream of "production" requests (sampled from golden cases, since we
    need SOME ground truth to spot-check against -- in reality you'd get
    ground truth from a small % of human-reviewed tickets, not all of it)
  - a rolling window of the last N predictions
  - three signals computed per window, each with its own threshold:
      1. spot-check accuracy   -- risk_level vs known ground truth, sampled
      2. output distribution shift -- risk_level mix vs the offline baseline
         (a real regression often shows up here BEFORE accuracy visibly drops,
         e.g. the model starts saying "Medium" for everything)
      3. p95 latency            -- slow responses are often an early symptom
         of a provider-side model swap or an overloaded endpoint
  - every prediction is appended to logs/predictions.jsonl (this is your
    audit trail -- what would flow into LangSmith / Vertex AI Model Monitoring
    / Evidently in a real system)
  - breaches are appended to logs/alerts.jsonl for respond_to_degradation.py

Usage:
    python production_monitor.py --requests 60 --inject-drift-after 30
"""

import argparse
import json
import os
import random
import statistics
from collections import Counter, deque
from datetime import datetime, timezone

from golden_dataset import GOLDEN_SET
from model import run_model

WINDOW_SIZE = 15
SPOT_CHECK_RATE = 0.3          # fraction of requests we have ground truth for
ACCURACY_ALERT_THRESHOLD = 0.75
DIST_SHIFT_ALERT_THRESHOLD = 0.35   # total variation distance vs baseline
LATENCY_ALERT_THRESHOLD_MS = 700

LOG_DIR = "logs"
PRED_LOG = os.path.join(LOG_DIR, "predictions.jsonl")
ALERT_LOG = os.path.join(LOG_DIR, "alerts.jsonl")

BASELINE_DIST = {"High": 5 / 10, "Medium": 3 / 10, "Low": 2 / 10}  # from golden set


def total_variation_distance(dist_a: dict, dist_b: dict) -> float:
    keys = set(dist_a) | set(dist_b)
    return 0.5 * sum(abs(dist_a.get(k, 0) - dist_b.get(k, 0)) for k in keys)


def append_jsonl(path, record):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def run_monitor(n_requests: int, inject_drift_after: int):
    # fresh logs each run so the demo is reproducible
    os.makedirs(LOG_DIR, exist_ok=True)
    open(PRED_LOG, "w").close()
    open(ALERT_LOG, "w").close()

    window = deque(maxlen=WINDOW_SIZE)
    alerts_raised = 0

    print(f"\n{'='*60}")
    print("PRODUCTION MONITOR -- production_monitor.py")
    print(f"Simulating {n_requests} requests, drift injected after request "
          f"#{inject_drift_after}")
    print(f"{'='*60}\n")

    for i in range(1, n_requests + 1):
        case = random.choice(GOLDEN_SET)
        drift = i > inject_drift_after
        spot_checked = random.random() < SPOT_CHECK_RATE

        pred = run_model(
            case["vehicle_id"], case["error_code"], case["sensor"], case["value"],
            history=case["history"], drift=drift,
        )
        correct = (pred["risk_level"] == case["expected_risk"]) if spot_checked else None

        record = {
            "seq": i,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_code": case["error_code"],
            "predicted_risk": pred["risk_level"],
            "expected_risk": case["expected_risk"] if spot_checked else None,
            "spot_checked": spot_checked,
            "correct": correct,
            "latency_ms": pred["latency_ms"],
            "drift_injected": drift,   # normally you would NOT know this in real life --
                                        # it's here only so the printed report can show
                                        # ground truth alongside what monitoring detected
        }
        append_jsonl(PRED_LOG, record)
        window.append(record)

        if len(window) == WINDOW_SIZE:
            checked = [w for w in window if w["spot_checked"]]
            window_acc = (sum(w["correct"] for w in checked) / len(checked)
                          if checked else None)
            dist = Counter(w["predicted_risk"] for w in window)
            total = sum(dist.values())
            window_dist = {k: v / total for k, v in dist.items()}
            shift = total_variation_distance(BASELINE_DIST, window_dist)
            p95 = sorted(w["latency_ms"] for w in window)[int(WINDOW_SIZE * 0.95) - 1]

            breaches = []
            if window_acc is not None and window_acc < ACCURACY_ALERT_THRESHOLD:
                breaches.append(f"spot-check accuracy {window_acc:.0%} < {ACCURACY_ALERT_THRESHOLD:.0%}")
            if shift > DIST_SHIFT_ALERT_THRESHOLD:
                breaches.append(f"output distribution shift {shift:.2f} > {DIST_SHIFT_ALERT_THRESHOLD}")
            if p95 > LATENCY_ALERT_THRESHOLD_MS:
                breaches.append(f"p95 latency {p95}ms > {LATENCY_ALERT_THRESHOLD_MS}ms")

            if breaches:
                alerts_raised += 1
                alert = {
                    "seq": i,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "window_accuracy": window_acc,
                    "window_distribution": window_dist,
                    "distribution_shift": round(shift, 3),
                    "p95_latency_ms": p95,
                    "breaches": breaches,
                }
                append_jsonl(ALERT_LOG, alert)
                print(f"  [ALERT] request #{i}: " + " | ".join(breaches))

    print(f"\n{n_requests} requests processed. {alerts_raised} alert window(s) raised.")
    if alerts_raised:
        print(f"See {ALERT_LOG} -- run respond_to_degradation.py next.")
    else:
        print("No degradation detected -- system healthy.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--inject-drift-after", type=int, default=30,
                         help="Set higher than --requests to simulate a fully healthy run.")
    args = parser.parse_args()
    run_monitor(args.requests, args.inject_drift_after)
