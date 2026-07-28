"""
PHASE 3 -- RESPONDING TO DEGRADATION

Detecting a problem (phase 2) is useless without a response. This script
reads logs/alerts.jsonl and logs/predictions.jsonl and carries out the
first few steps of PLAYBOOK.md automatically -- the steps a human would
otherwise have to do by hand at 2am.

What it does, in order:
  1. Freeze:        writes deploy_frozen.flag -- a real CI/CD pipeline would
                     check for this file and refuse to promote new prompt/
                     model versions until it's cleared.
  2. Quarantine:     pulls the predictions around the first alert into
                     review_queue.json for a human to grade.
  3. Diagnose:       looks at WHICH signal broke (accuracy vs distribution
                     shift vs latency) to suggest a likely cause.
  4. Record:         appends a timestamped entry to incidents.md so there's
                     a paper trail (this is what you'd hand to a postmortem).

Usage:
    python respond_to_degradation.py
"""

import json
import os
from datetime import datetime, timezone

LOG_DIR = "logs"
ALERT_LOG = os.path.join(LOG_DIR, "alerts.jsonl")
PRED_LOG = os.path.join(LOG_DIR, "predictions.jsonl")
FREEZE_FLAG = "deploy_frozen.flag"
REVIEW_QUEUE = "review_queue.json"
INCIDENTS = "incidents.md"


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def suggest_cause(breaches: list[str]) -> str:
    text = " ".join(breaches)
    if "distribution shift" in text and "accuracy" not in text:
        return ("Output distribution shifted before accuracy visibly dropped. "
                "Classic early-warning sign: the model/prompt is drifting toward "
                "one label (often 'Medium' as a safe default) before it starts "
                "failing spot-checks outright. Check for: silent model version "
                "change on the provider side, an unreviewed prompt edit, or a "
                "new input pattern the prompt wasn't written for.")
    if "latency" in text and "accuracy" not in text:
        return ("Latency degraded without an accuracy signal yet. Check for: "
                "provider-side incident, endpoint overload, or a fallback/retry "
                "path silently kicking in. Not necessarily a quality regression "
                "-- but often a leading indicator of one, since providers "
                "sometimes reroute traffic to a smaller model under load.")
    if "accuracy" in text:
        return ("Spot-check accuracy dropped below threshold -- this is the "
                "clearest and most serious signal. Treat as a confirmed quality "
                "regression, not a false alarm.")
    return "Multiple signals breached simultaneously -- treat as high-confidence regression."


def respond():
    alerts = read_jsonl(ALERT_LOG)
    if not alerts:
        print("No alerts found in logs/alerts.jsonl. Nothing to respond to.")
        print("(Run production_monitor.py with drift injected first.)")
        return

    predictions = read_jsonl(PRED_LOG)
    first_alert = alerts[0]
    window_start = max(1, first_alert["seq"] - 15)
    flagged = [p for p in predictions if window_start <= p["seq"] <= first_alert["seq"]]

    # 1. Freeze deploys
    with open(FREEZE_FLAG, "w") as f:
        f.write(json.dumps({
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "reason": f"Alert raised at request #{first_alert['seq']}: {first_alert['breaches']}",
        }, indent=2))
    print(f"[1/4] FREEZE  -> wrote {FREEZE_FLAG}. New deploys should be blocked "
          f"until this is cleared.")

    # 2. Quarantine flagged predictions for human review
    with open(REVIEW_QUEUE, "w") as f:
        json.dump(flagged, f, indent=2)
    print(f"[2/4] REVIEW  -> wrote {REVIEW_QUEUE} ({len(flagged)} predictions "
          f"around the first alert) for human grading.")

    # 3. Suggest likely cause
    cause = suggest_cause(first_alert["breaches"])
    print(f"[3/4] DIAGNOSE -> {cause}")

    # 4. Record incident
    entry = f"""
## Incident -- {datetime.now(timezone.utc).isoformat()}

- **Triggered at request**: #{first_alert['seq']}
- **Breaches**: {"; ".join(first_alert['breaches'])}
- **Window accuracy**: {first_alert['window_accuracy']}
- **Output distribution**: {first_alert['window_distribution']}
- **Distribution shift (TVD)**: {first_alert['distribution_shift']}
- **P95 latency**: {first_alert['p95_latency_ms']}ms
- **Likely cause**: {cause}
- **Actions taken**: deploy frozen ({FREEZE_FLAG}), {len(flagged)} predictions
  queued for human review ({REVIEW_QUEUE})
- **Next steps** (manual, see PLAYBOOK.md):
  1. Human reviewer grades {REVIEW_QUEUE} against golden rubric
  2. If confirmed regression: roll back to last known-good prompt/model version
  3. If it's a genuinely new input pattern: add it to golden_dataset.py so
     offline_eval.py catches this class of case next time
  4. Re-run offline_eval.py against the rollback candidate before re-enabling
     deploys (delete {FREEZE_FLAG} only after it passes)
"""
    with open(INCIDENTS, "a") as f:
        f.write(entry)
    print(f"[4/4] RECORD  -> appended incident to {INCIDENTS}")

    print(f"\n{len(alerts)} total alert window(s) in this run. "
          f"Responded to the first; see {INCIDENTS} for the full record.")


if __name__ == "__main__":
    respond()
