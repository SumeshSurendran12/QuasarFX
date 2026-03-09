export type HealthStatus = "HEALTHY" | "ATTENTION" | "CRITICAL" | string;

export interface DailyHealthCheck {
  name: string;
  pass: boolean;
}

export interface DailyHealthHeartbeat {
  ingestion_state: string;
  events_total: number;
  source_events_total: number;
  source_events_total_unfiltered: number;
  source_events_excluded_before_window: number;
  first_event_ts_utc: string;
  last_event_ts_utc: string;
  process_start_count: number;
  last_process_start_ts_utc: string;
  last_process_start_age_minutes: number;
}

export interface PaperSnapshot {
  paper_status: string;
  paper_days: number;
  trade_count: number;
  net_pnl_usd: number;
  profit_factor: number;
  last_trade_age_minutes: number;
}

export interface DailyHealthReport {
  status: HealthStatus;
  canonical_window_start_utc: string;
  heartbeat: DailyHealthHeartbeat;
  checks: DailyHealthCheck[];
  paper_snapshot: PaperSnapshot;
}

export interface PaperReportSummary {
  trade_count: number;
  wins: number;
  losses: number;
  net_pnl_usd: number;
  gross_profit_usd: number;
  gross_loss_usd: number;
  profit_factor: number;
  max_drawdown_usd: number;
  spread_gate_skips: number;
  session_cap_skips: number;
  event_counts: Record<string, number>;
  skip_reason_counts: Record<string, number>;
  contract_violation_counters: Record<string, number>;
  daily_pnl: Record<string, number>;
  contract_violation_counters_by_day: Record<string, Record<string, number>>;
}

export interface PaperReport {
  status: string;
  summary: PaperReportSummary;
}

export interface PnlRow {
  date: string;
  pnl: number;
  cumulative: number;
}

export interface SkipRow {
  name: string;
  value: number;
}

export interface IntegrityRow {
  date: string;
  total: number;
  [key: string]: number | string;
}

export interface JsonParseResult<T> {
  data: T | null;
  error: string | null;
}
