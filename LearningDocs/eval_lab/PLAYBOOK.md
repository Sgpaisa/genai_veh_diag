# LLM Evaluation & Degradation Playbook

This is the decision framework `eval_lab/` demonstrates in miniature. It applies
to any LLM feature in production, not just this vehicle diagnostics project.

## The three phases

| Phase | Question it answers | When it runs | Script here | Real-world tool |
|---|---|---|---|---|
| 1. Offline eval | "Is this new prompt/model version good enough to ship?" | Every code/prompt/model change, in CI | `offline_eval.py` | LangSmith datasets, promptfoo, Braintrust |
| 2. Online monitoring | "Is the thing I already shipped still good?" | Continuously, on live traffic | `production_monitor.py` | Vertex AI Model Monitoring, LangSmith tracing, Evidently, Arize |
| 3. Incident response | "It broke -- now what?" | Triggered by an alert | `respond_to_degradation.py` | PagerDuty + a runbook, feature flags, CI gate |

## Why you need both offline eval AND online monitoring

Offline eval only proves the model is good on the cases you thought to write
down. It cannot catch:
- New input patterns you didn't anticipate
- Silent provider-side model swaps (e.g. `gemini-2.5-flash` pointing at a
  retrained checkpoint with the same name)
- Prompt edits that skipped the gate (someone hotfixes a typo directly)
- Slow distribution drift in real-world inputs (fleet starts reporting new
  error codes as vehicles age)

Online monitoring only works if you have SOME ground truth to compare
against. In practice that means:
- Spot-checking a sample of production outputs with human reviewers
- Using downstream signals as weak proxies (did the mechanic override the
  AI's recommendation? did the "High risk" vehicle actually break down?)
- Comparing the output *distribution* against a known-good baseline, even
  without per-case ground truth (this is what `production_monitor.py`'s
  distribution-shift check does)

## What to actually watch in production

1. **Hard-label accuracy on spot-checks** (`risk_level` here). This is the
   safety-critical field -- weight it most heavily and alert early.
2. **Output distribution drift** vs. a known baseline. Often the earliest
   signal: a degraded model tends to collapse toward one "safe" answer
   before it starts failing spot-checks outright.
3. **Latency / cost**. Sudden latency spikes often precede quality issues
   (provider incident, silent fallback to a smaller model, retries masking
   failures).
4. **Schema/parse failure rate**. If you're using `response_schema` (as
   `diagnostics.py` does), track how often parsing fails -- a rising rate
   means the model is drifting away from the format you asked for.
5. **Judge score on free text** (`summary`, `root_causes`). Needs an
   LLM-as-judge or periodic human grading since you can't `==` a paragraph.

## Thresholds used in this lab (tune per your risk tolerance)

| Signal | Threshold | Rationale |
|---|---|---|
| Risk accuracy (offline gate) | ≥ 90% | Safety-critical field, near-zero tolerance |
| Judge score (offline gate) | ≥ 3.5 / 5 | Explanation quality, some tolerance for style variance |
| Spot-check accuracy (online) | ≥ 75% | Lower than offline because online sample is noisier/smaller |
| Distribution shift (online) | ≤ 0.35 TVD | Catches collapse-toward-one-label before accuracy visibly drops |
| P95 latency (online) | ≤ 700ms | Early proxy for provider-side issues |

## When an alert fires: the response sequence

1. **Freeze** new deploys/prompt changes until the incident is resolved.
   Don't let a second, unrelated change land on top of a live regression --
   you'll never isolate the cause.
2. **Quarantine** the flagged window of predictions for human review. Don't
   trust the judge/monitor's own verdict as the final word -- it can be wrong
   or itself be drifting.
3. **Diagnose** which signal broke first (accuracy vs. distribution vs.
   latency) -- it tells you where to look (prompt vs. provider vs. infra).
4. **Decide**:
   - Confirmed regression -> **roll back** to the last version that passed
     the offline gate. Don't try to "prompt-engineer around it" live.
   - New legitimate input pattern, not a regression -> **add it to the
     golden dataset** so offline_eval.py catches this class of case for
     every future change (this is how your eval set should grow over time).
   - Judge/monitor false positive -> tune the threshold, but log why, don't
     just silence the alert.
5. **Re-verify** the rollback (or fix) against `offline_eval.py` before
   unfreezing deploys. Never unfreeze on "it looks fine now" -- rerun the gate.
6. **Postmortem**: `incidents.md` is the seed of this. What golden case would
   have caught this earlier? Add it.

## Mapping to this codebase specifically

- `diagnostics.py` is the production model call this lab's `model.py`
  mimics -- same `DiagnosisResult` schema, same `temperature=0.1` convention.
- `run_evals.py` is a real offline-eval example (LangSmith-based, exact-match
  only). `offline_eval.py` here extends that pattern with a judge score for
  free text -- consider porting that addition back into `run_evals.py`.
- In a real deployment, `production_monitor.py`'s job would be done by
  LangSmith tracing (since `run_evals.py` already uses LangSmith) or Vertex
  AI Model Monitoring, wired into `api.py`'s `/diagnose` endpoint so every
  real request gets logged, not just simulated ones.
- The `deploy_frozen.flag` idea would become an actual CI/CD gate check --
  e.g. a step in `.github/` workflows that fails the build if the flag file
  (or an equivalent flag in a feature-flag service) is present.
