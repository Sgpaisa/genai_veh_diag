# eval_lab -- LLM Evaluation Before & After Production

A small, self-contained project to learn one thing clearly: how do you know
an LLM feature is good enough to ship, and how do you know if it silently
breaks after it's live?

Uses the same domain as the parent project (OBD-II risk diagnosis) so the
concepts map directly onto `diagnostics.py` / `run_evals.py`, but runs
standalone with **no GCP setup required** -- a mock model backend stands in
for Gemini so you can run the whole thing in under a minute.

Read `PLAYBOOK.md` first for the concepts. Then run the three phases below
to see them in action.

## Setup

```bash
cd eval_lab
pip install python-dotenv   # only real dependency for mock mode
```

No `.env` needed to run in mock mode. If you want to run against the real
Gemini API instead, add `GEMINI_API_KEY=...` to a `.env` file in this folder
and install `google-genai` + `pydantic` -- `model.py` and `judge.py` will
switch to real calls automatically.

## Phase 1 -- Offline eval (the pre-deploy gate)

```bash
python offline_eval.py
```

Runs the model against 10 hand-labeled golden cases, checks exact-match
accuracy on `risk_level` and an LLM-as-judge score on the free-text
explanation. Prints PASS/FAIL and writes `offline_eval_report.json`.

Try it with a "bad" version to see it fail — edit the call at the bottom of
`offline_eval.py` to `evaluate(drift=True)` and rerun.

## Phase 2 -- Production monitoring (continuous, post-deploy)

```bash
python production_monitor.py --requests 60 --inject-drift-after 30
```

Simulates 60 production requests. For the first 30, the model behaves well
(matches phase 1). After request 30, drift is injected -- the mock model
starts giving vaguer, occasionally-wrong-and-understated risk levels with
higher latency, standing in for a real-world regression (bad prompt edit,
provider-side model swap, etc).

Watch for `[ALERT]` lines. Everything is also logged to `logs/predictions.jsonl`
and `logs/alerts.jsonl`.

Run it again with `--inject-drift-after 1000` (never triggers, since only 60
requests run) to see what a healthy, no-alert run looks like.

## Phase 3 -- Respond to degradation

```bash
python respond_to_degradation.py
```

Reads the alerts from phase 2 and runs the first steps of the playbook:
freezes deploys (`deploy_frozen.flag`), quarantines the flagged predictions
for human review (`review_queue.json`), suggests a likely cause based on
which signal broke first, and records the incident (`incidents.md`).

## Suggested order to actually learn this

1. Read `PLAYBOOK.md` top to bottom.
2. Run phase 1 clean (`drift=False`) -- see it pass.
3. Run phase 1 with `drift=True` -- see it fail, read why.
4. Run phase 2 with drift injected -- watch alerts fire, open
   `logs/alerts.jsonl` and look at the distribution-shift numbers.
5. Run phase 3 -- read `incidents.md` and `review_queue.json`.
6. Re-read the "Mapping to this codebase specifically" section of
   `PLAYBOOK.md` and decide what you'd actually wire into `api.py` /
   `run_evals.py` for the real project.

## Files

| File | Role |
|---|---|
| `golden_dataset.py` | Hand-labeled ground truth for offline eval |
| `model.py` | Model under test (mock or real Gemini) |
| `judge.py` | LLM-as-judge for free-text quality (mock or real Gemini) |
| `offline_eval.py` | Phase 1: pre-deploy gate |
| `production_monitor.py` | Phase 2: simulated live traffic + rolling alerts |
| `respond_to_degradation.py` | Phase 3: automated first-response steps |
| `PLAYBOOK.md` | The concepts and decision framework |
| `logs/predictions.jsonl` | Every simulated production prediction (generated) |
| `logs/alerts.jsonl` | Every threshold breach (generated) |
| `incidents.md` | Incident record (generated, appended to) |
