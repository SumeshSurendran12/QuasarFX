# Strategy 1 Deployment Checklist

Reference runbook: `docs/strategy_1_promotion_runbook.md`
Daily status log: `docs/strategy_1_daily_status_log.md`

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
  - required common fields: `ts`, `run_id`, `event`, `stage`, `strategy_id`, `profile_hash`, `schema_version`, `manifest_version`
  - run_id pattern: `YYYY-MM-DD_SESSION_shaXXXXXXXX`
  - reason_code vocabulary is immutable

## Paper Validation Window
- Run paper mode for at least 30 calendar days
- Minimum 40 trades in the validation window
- No kill-switch breaches
- Daily health report generated and reviewed

## Logging and Alerting
- Canonical execution log enabled (`events.jsonl`)
- End-of-day aggregate generated (`daily_summary.json`)
- Dated audit artifacts generated under `reports/YYYY-MM-DD/`
- Spread-gate skips logged
- API failures logged
- Profile hash logged on every event
- schema_version + manifest_version logged on every event
- Daily health report generated (`scripts/daily_health_report.py`)
- Daily pipeline scheduled (`scripts/run_daily_paper_pipeline.ps1`) and runs summary -> paper -> health in that order

## Fail-Safe No-Trade Conditions
- Spread too high
- Too many skipped bars
- Broker/API failure burst
- Daily drawdown breach

## Promotion Gate to LIVE_GATED
### Integrity Pass
- Event contract checks pass (schema/hash/version/run_id)
- Monotonic event order check passes
- Trade lifecycle completeness check passes
- Daily summary reconciliation check passes
- Process-start metadata present on current paper sessions (`process_start_ts`)
- Restart frequency check passes (`restart_frequency_healthy`)

### Policy Pass
- No kill-switch breaches
- Spread gate and session cap behavior within policy
- Frozen profile unchanged

### Performance Pass
- Paper window requirements met
- Controls and kill-switch checks pass
- Minimum trade count met
- Performance metrics within tolerated band
- Final review sign-off recorded
