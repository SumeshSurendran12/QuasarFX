import type { DailyHealthReport, PaperReport } from "./types";

export const sampleDailyHealth: DailyHealthReport = {
  status: "ATTENTION",
  canonical_window_start_utc: "2026-03-08T04:32:49+00:00",
  heartbeat: {
    ingestion_state: "ACTIVE",
    events_total: 1019,
    source_events_total: 1129,
    source_events_total_unfiltered: 1129,
    source_events_excluded_before_window: 110,
    first_event_ts_utc: "2026-03-08T22:47:56.550392Z",
    last_event_ts_utc: "2026-03-08T22:49:25.919201Z",
    process_start_count: 1,
    last_process_start_ts_utc: "2026-03-08T22:47:56.550392Z",
    last_process_start_age_minutes: 0
  },
  checks: [
    { name: "daily_summary_reconciled", pass: true },
    { name: "trade_lifecycle_complete", pass: true },
    { name: "process_start_metadata_present", pass: true },
    { name: "restart_frequency_healthy", pass: true },
    { name: "paper_window_pass", pass: false }
  ],
  paper_snapshot: {
    paper_status: "PAPER_IN_PROGRESS",
    paper_days: 1,
    trade_count: 1,
    net_pnl_usd: 12.4,
    profit_factor: 1.8,
    last_trade_age_minutes: 3
  }
};

export const samplePaperReport: PaperReport = {
  status: "PAPER_IN_PROGRESS",
  summary: {
    trade_count: 1,
    wins: 1,
    losses: 0,
    net_pnl_usd: 12.4,
    gross_profit_usd: 12.4,
    gross_loss_usd: 0,
    profit_factor: 1.8,
    max_drawdown_usd: 0,
    spread_gate_skips: 410,
    session_cap_skips: 95,
    event_counts: {
      session_start: 1,
      data_feed_alive: 2,
      signal_evaluated: 507,
      trade_skipped: 505,
      order_submitted: 1,
      order_filled: 1,
      position_closed: 1,
      session_end: 1
    },
    skip_reason_counts: {
      spread_gate: 410,
      session_cap: 95
    },
    contract_violation_counters: {
      schema_version_mismatch: 0,
      manifest_version_mismatch: 0,
      invalid_run_id: 0,
      run_id_hash_prefix_mismatch: 0,
      profile_hash_mismatch: 0,
      profile_hash_missing: 0,
      monotonic_order_violations: 0,
      lifecycle_fill_without_submit: 0,
      lifecycle_close_without_fill: 0,
      lifecycle_duplicate_close: 0,
      lifecycle_mixed_side: 0
    },
    daily_pnl: {
      "2026-03-08": 12.4,
      "2026-03-09": -3.2,
      "2026-03-10": 6.8,
      "2026-03-11": 4.5,
      "2026-03-12": -1.7,
      "2026-03-13": 8.1
    },
    contract_violation_counters_by_day: {
      "2026-03-08": {
        schema_version_mismatch: 0,
        manifest_version_mismatch: 0,
        invalid_run_id: 0,
        run_id_hash_prefix_mismatch: 0,
        profile_hash_mismatch: 0,
        profile_hash_missing: 0,
        monotonic_order_violations: 0,
        lifecycle_fill_without_submit: 0,
        lifecycle_close_without_fill: 0,
        lifecycle_duplicate_close: 0,
        lifecycle_mixed_side: 0
      },
      "2026-03-09": {
        schema_version_mismatch: 0,
        manifest_version_mismatch: 0,
        invalid_run_id: 0,
        run_id_hash_prefix_mismatch: 0,
        profile_hash_mismatch: 0,
        profile_hash_missing: 0,
        monotonic_order_violations: 0,
        lifecycle_fill_without_submit: 0,
        lifecycle_close_without_fill: 0,
        lifecycle_duplicate_close: 0,
        lifecycle_mixed_side: 0
      },
      "2026-03-10": {
        schema_version_mismatch: 0,
        manifest_version_mismatch: 0,
        invalid_run_id: 1,
        run_id_hash_prefix_mismatch: 0,
        profile_hash_mismatch: 0,
        profile_hash_missing: 0,
        monotonic_order_violations: 0,
        lifecycle_fill_without_submit: 0,
        lifecycle_close_without_fill: 0,
        lifecycle_duplicate_close: 0,
        lifecycle_mixed_side: 0
      },
      "2026-03-11": {
        schema_version_mismatch: 0,
        manifest_version_mismatch: 0,
        invalid_run_id: 0,
        run_id_hash_prefix_mismatch: 0,
        profile_hash_mismatch: 0,
        profile_hash_missing: 0,
        monotonic_order_violations: 0,
        lifecycle_fill_without_submit: 0,
        lifecycle_close_without_fill: 0,
        lifecycle_duplicate_close: 0,
        lifecycle_mixed_side: 0
      },
      "2026-03-12": {
        schema_version_mismatch: 0,
        manifest_version_mismatch: 0,
        invalid_run_id: 0,
        run_id_hash_prefix_mismatch: 0,
        profile_hash_mismatch: 0,
        profile_hash_missing: 0,
        monotonic_order_violations: 0,
        lifecycle_fill_without_submit: 0,
        lifecycle_close_without_fill: 0,
        lifecycle_duplicate_close: 1,
        lifecycle_mixed_side: 0
      },
      "2026-03-13": {
        schema_version_mismatch: 0,
        manifest_version_mismatch: 0,
        invalid_run_id: 0,
        run_id_hash_prefix_mismatch: 0,
        profile_hash_mismatch: 0,
        profile_hash_missing: 0,
        monotonic_order_violations: 0,
        lifecycle_fill_without_submit: 0,
        lifecycle_close_without_fill: 0,
        lifecycle_duplicate_close: 0,
        lifecycle_mixed_side: 0
      }
    }
  }
};
