# Kill-Switch Policy (Strategy 1)

## Scope
- Strategy: `Strategy 1 - London Compression Breakout`
- Stage: `PAPER_CANDIDATE` (promote to `LIVE_GATED` only after paper checks pass)

## Trigger Conditions
1. Spread too high
- Trigger: any entry attempt with spread `> 0.00022`
- Action: block new entries; keep monitoring; alert operator

2. Too many skipped bars
- Trigger: consecutive skipped bars `> 120`
- Action: stop new entries for the session; alert operator

3. Broker/API failure
- Trigger: consecutive broker/API failures `>= 3`
- Action: stop trading loop; alert operator; require manual restart

4. Daily drawdown breach
- Trigger: daily realized PnL `<= -120.0 USD`
- Action: halt trading for the remainder of the day; alert operator

## Recovery Rules
- Recovery from spread and skip-bar triggers is automatic only after trigger condition clears.
- Recovery from broker/API failure and drawdown breach is manual.
- Every halt must log trigger reason, timestamp, and last known state.

## Required Telemetry
- Entry spread and spread-gate block count
- Per-session trade count
- Consecutive skipped bars
- Consecutive API failures
- Daily realized PnL and drawdown
- Kill-switch state (`ACTIVE` / `TRIGGERED`)
