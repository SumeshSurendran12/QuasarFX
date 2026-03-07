# Strategy 1 Deployment Checklist

## Stage Labels
- Strategy 1: `PAPER_CANDIDATE`
- Strategy 2 deterministic branches: `RESEARCH`
- RLM/RL branches: `EXPERIMENTAL_ONLY`
- Promotion target after paper checks: `LIVE_GATED`

## Frozen Profile
- `strategy_1_profile.json` exists and is versioned
- Frozen parameters are unchanged during the paper-validation window
- Live controls are explicit:
  - spread gate active
  - one trade/session cap

## Paper Validation Window
- Run paper mode for at least 30 calendar days
- Minimum 40 trades in the validation window
- No kill-switch breaches
- Daily health report generated and reviewed

## Logging and Alerting
- Action/decision logs enabled
- Spread-gate skips logged
- API failures logged
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
