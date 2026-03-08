# Strategy 1 Promotion Runbook

## Lifecycle
`RESEARCH -> PAPER_CANDIDATE -> LIVE_GATED -> PRODUCTION`

## Phase 1: Paper Validation Window
- Keep `strategy_1` in `PAPER_CANDIDATE`.
- Do not change strategy logic or frozen parameters during the paper window.
- Required minimums:
  - `paper_days >= 30`
  - `trade_count >= 40`

### Daily Routine
1. Run paper session:
   - `python modules/live_trading.py`
2. Run daily pipeline:
   - `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_daily_paper_pipeline.ps1`
3. Review:
   - `reports/YYYY-MM-DD/daily_health.json`
   - `reports/YYYY-MM-DD/paper_report.json`

### Daily Integrity Requirements
- `trade_lifecycle_complete = true`
- `daily_summary_reconciled = true`
- `process_start_metadata_present = true`
- `restart_frequency_healthy = true`

### Daily Policy Requirements
- `spread_gate_active = true`
- `one_trade_per_session_cap = true`
- `kill_switch_clear = true`

Expected until minimum window is met:
- `status = ATTENTION`
- `paper_window_pass = false`

## Weekly Integrity Review
- Use this checklist:

| Category | What to check |
| --- | --- |
| Integrity | Lifecycle errors remain `0` (no duplicate close, no fill/close corruption). |
| Contract | Schema/contract violations remain `0`. |
| Restart stability | `restart_frequency_healthy = true`. |
| Reconciliation | `daily_summary_reconciled = true` for all runs. |
| Skip distribution | Skip mix is stable (no unexpected drift in reason distribution). |
| Event ingestion | No silent gaps in canonical events. |
| Run ID uniqueness | No `run_id` collisions. |

If any integrity requirement breaks, restart the formal paper window.

## Daily Status Log
- Keep one line per day in `docs/strategy_1_daily_status_log.md`.
- Format:
  - `YYYY-MM-DD | events=<n> | trades=<n> | status=<status> | integrity=<PASS|FAIL> | notes=<short note>`

## Promotion Review Gate
All three gates must pass.

### Integrity Pass
- Reconciliation and lifecycle checks pass.
- Restart and process-start checks pass.
- Contract violations are zero.

### Policy Pass
- Kill switch clear.
- Spread gate and one-trade/session cap enforced.
- Frozen profile unchanged.

### Performance Pass
- `paper_days >= 30`
- `trade_count >= 40`
- Profit factor acceptable (typical target: `> 1.2`)
- Net PnL positive.
- Drawdown within tolerance.
- Spread-skip behavior stable.

## Phase 2: LIVE_GATED
- Promote `strategy_1` label to `LIVE_GATED` after gate approval.
- Use strict limits (example):
  - `risk_per_trade = 0.25R`
  - conservative daily loss limit
  - `max_trades_per_session = 1`
- Observe for a live-gated window (typical: `7-14` trading days).
- Monitor:
  - slippage
  - API reliability
  - kill-switch triggers
  - reconciliation stability

## Phase 3: PRODUCTION
- Promote `strategy_1` to `PRODUCTION` only if live-gated window is clean.
- Keep the same daily monitoring pipeline active:
  - `daily_summary`
  - `paper_report`
  - `health_report`
