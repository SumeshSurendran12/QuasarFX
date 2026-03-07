# Strategy 1 Deployment Checklist

## Stage Labels
- Strategy 1: `PAPER_CANDIDATE`
- Strategy 2 deterministic branches: `RESEARCH`
- RLM/RL branches: `EXPERIMENTAL_ONLY`
- Promotion target after paper checks: `LIVE_GATED`

## Frozen Profile
- `strategy_1_profile.json` exists and is versioned
- `manifest.json` exists and is versioned
- Frozen parameters are unchanged during the paper-validation window
- Live controls are explicit:
  - spread gate active
  - one trade/session cap
- Event contract is frozen:
  - schema: `schemas/strategy_1_events.schema.json`
  - required common fields: `ts`, `run_id`, `event`, `stage`, `strategy_id`, `profile_hash`

## Paper Validation Window
- Run paper mode for at least 30 calendar days
- Minimum 40 trades in the validation window
- No kill-switch breaches
- Daily health report generated and reviewed

## Logging and Alerting
- Canonical execution log enabled (`events.jsonl`)
- End-of-day aggregate generated (`daily_summary.json`)
- Spread-gate skips logged
- API failures logged
- Profile hash logged on every event
- Daily health report generated (`scripts/daily_health_report.py`)

## Fail-Safe No-Trade Conditions
- Spread too high
- Too many skipped bars
- Broker/API failure burst
- Daily drawdown breach

## Promotion Gate to LIVE_GATED
- Paper window requirements met
- Controls and kill-switch checks pass
- Frozen profile unchanged
- Final review sign-off recorded
