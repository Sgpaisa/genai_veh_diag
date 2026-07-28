
## Incident -- 2026-07-28T06:43:46.817963+00:00

- **Triggered at request**: #32
- **Breaches**: p95 latency 791ms > 700ms
- **Window accuracy**: 1.0
- **Output distribution**: {'Medium': 0.3333333333333333, 'High': 0.4, 'Low': 0.26666666666666666}
- **Distribution shift (TVD)**: 0.1
- **P95 latency**: 791ms
- **Likely cause**: Latency degraded without an accuracy signal yet. Check for: provider-side incident, endpoint overload, or a fallback/retry path silently kicking in. Not necessarily a quality regression -- but often a leading indicator of one, since providers sometimes reroute traffic to a smaller model under load.
- **Actions taken**: deploy frozen (deploy_frozen.flag), 16 predictions
  queued for human review (review_queue.json)
- **Next steps** (manual, see PLAYBOOK.md):
  1. Human reviewer grades review_queue.json against golden rubric
  2. If confirmed regression: roll back to last known-good prompt/model version
  3. If it's a genuinely new input pattern: add it to golden_dataset.py so
     offline_eval.py catches this class of case next time
  4. Re-run offline_eval.py against the rollback candidate before re-enabling
     deploys (delete deploy_frozen.flag only after it passes)
