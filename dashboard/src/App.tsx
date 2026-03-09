import { useMemo, useState } from "react";
import type { ChangeEvent, Dispatch, SetStateAction } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { Activity, AlertTriangle, CheckCircle2, ShieldCheck, Upload } from "lucide-react";
import type { DailyHealthReport, IntegrityRow, PaperReport, PaperReportSummary, PnlRow, SkipRow } from "./types";
import { parseJson } from "./json";
import { sampleDailyHealth, samplePaperReport } from "./sampleData";

type RangeValue = "7d" | "14d" | "all";

function statusTone(status: string): string {
  if (status === "HEALTHY") return "badge healthy";
  if (status === "ATTENTION") return "badge attention";
  return "badge critical";
}

function formatCurrency(value: number): string {
  return `$${value.toFixed(2)}`;
}

function formatProfitFactor(summary?: PaperReportSummary): string {
  if (!summary) return "N/A";
  const grossProfit = Number(summary.gross_profit_usd) || 0;
  const grossLoss = Number(summary.gross_loss_usd) || 0;
  if (grossLoss === 0 && grossProfit > 0) return "∞";

  const pf = Number(summary.profit_factor);
  return Number.isFinite(pf) ? pf.toFixed(2) : "N/A";
}

function sortByDateKey<T>(rows: T[], getDate: (row: T) => string): T[] {
  return [...rows].sort((a, b) => getDate(a).localeCompare(getDate(b)));
}

function handleFileLoad(
  event: ChangeEvent<HTMLInputElement>,
  setter: Dispatch<SetStateAction<string>>
): void {
  const file = event.target.files?.[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = () => setter(String(reader.result ?? ""));
  reader.readAsText(file);
  event.target.value = "";
}

export default function App() {
  const [healthText, setHealthText] = useState<string>("");
  const [paperText, setPaperText] = useState<string>("");
  const [range, setRange] = useState<RangeValue>("all");

  const healthParse = useMemo(() => parseJson<DailyHealthReport>(healthText), [healthText]);
  const paperParse = useMemo(() => parseJson<PaperReport>(paperText), [paperText]);
  const health = healthParse.data;
  const paper = paperParse.data;

  const checkSummary = useMemo(() => {
    const checks = health?.checks ?? [];
    const pass = checks.filter((check) => check.pass).length;
    const fail = checks.filter((check) => !check.pass).length;
    return { pass, fail, total: checks.length };
  }, [health]);

  const pnlSeries = useMemo<PnlRow[]>(() => {
    const rows: PnlRow[] = Object.entries(paper?.summary.daily_pnl ?? {}).map(([date, pnl]) => ({
      date,
      pnl: Number(pnl) || 0,
      cumulative: 0
    }));
    const sorted = sortByDateKey(rows, (row) => row.date);

    let cumulative = 0;
    const withCumulative = sorted.map((row) => {
      cumulative += row.pnl;
      return { ...row, cumulative: Number(cumulative.toFixed(2)) };
    });

    if (range === "7d") return withCumulative.slice(-7);
    if (range === "14d") return withCumulative.slice(-14);
    return withCumulative;
  }, [paper, range]);

  const skipSeries = useMemo<SkipRow[]>(() => {
    return Object.entries(paper?.summary.skip_reason_counts ?? {})
      .map(([name, value]) => ({ name, value: Number(value) || 0 }))
      .sort((a, b) => b.value - a.value);
  }, [paper]);

  const integritySeries = useMemo<IntegrityRow[]>(() => {
    const rows: IntegrityRow[] = Object.entries(paper?.summary.contract_violation_counters_by_day ?? {}).map(
      ([date, counts]) => {
        const total = Object.values(counts ?? {}).reduce((acc, value) => acc + (Number(value) || 0), 0);
        return { date, total, ...counts };
      }
    );
    return sortByDateKey(rows, (row) => row.date);
  }, [paper]);

  const integrityAlerts = useMemo(() => {
    return (health?.checks ?? []).filter((check) => !check.pass).map((check) => check.name);
  }, [health]);

  const topMetrics = useMemo(() => {
    return [
      {
        title: "Status",
        value: health?.status ?? "UNKNOWN",
        subtitle: `Paper status: ${health?.paper_snapshot.paper_status ?? "N/A"}`
      },
      {
        title: "Paper days",
        value: String(health?.paper_snapshot.paper_days ?? 0),
        subtitle: `Trades: ${health?.paper_snapshot.trade_count ?? 0}`
      },
      {
        title: "Net PnL",
        value: formatCurrency(Number(health?.paper_snapshot.net_pnl_usd ?? 0)),
        subtitle: `PF: ${formatProfitFactor(paper?.summary)}`
      },
      {
        title: "Events",
        value: String(health?.heartbeat.events_total ?? 0),
        subtitle: `Excluded before window: ${health?.heartbeat.source_events_excluded_before_window ?? 0}`
      },
      {
        title: "Restarts",
        value: String(health?.heartbeat.process_start_count ?? 0),
        subtitle: `Last start age: ${health?.heartbeat.last_process_start_age_minutes ?? "n/a"} min`
      },
      {
        title: "Checks",
        value: `${checkSummary.pass}/${checkSummary.total}`,
        subtitle: `${checkSummary.fail} failing`
      }
    ];
  }, [health, paper, checkSummary]);

  const loadDemoData = (): void => {
    setHealthText(JSON.stringify(sampleDailyHealth, null, 2));
    setPaperText(JSON.stringify(samplePaperReport, null, 2));
  };

  const clearInputs = (): void => {
    setHealthText("");
    setPaperText("");
  };

  return (
    <div className="page">
      <div className="container">
        <header className="header">
          <div>
            <div className="title-wrap">
              <ShieldCheck className="icon-lg" />
              <h1>Strategy 1 Monitoring Dashboard</h1>
            </div>
            <p>
              Weekly review view from <code>daily_health.json</code> and <code>paper_report.json</code>.
            </p>
          </div>
          <div className="header-controls">
            <span className={statusTone(health?.status ?? "UNKNOWN")}>{health?.status ?? "UNKNOWN"}</span>
            <div className="tabs">
              {(["7d", "14d", "all"] as RangeValue[]).map((value) => (
                <button
                  key={value}
                  className={range === value ? "tab active" : "tab"}
                  onClick={() => setRange(value)}
                  type="button"
                >
                  {value.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
        </header>

        <div className="metrics-grid">
          {topMetrics.map((metric) => (
            <article key={metric.title} className="card metric">
              <div className="metric-title">{metric.title}</div>
              <div className="metric-value">{metric.value}</div>
              <div className="metric-subtitle">{metric.subtitle}</div>
            </article>
          ))}
        </div>

        <section className="grid-main">
          <article className="card panel-wide">
            <h2>
              <Activity className="icon-md" /> PnL Curve
            </h2>
            <p>Daily and cumulative PnL across the canonical validation window.</p>
            <div className="chart-box">
              {pnlSeries.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={pnlSeries}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Area type="monotone" dataKey="pnl" name="Daily PnL" stroke="#f97316" fill="#fb923c" fillOpacity={0.22} />
                    <Line type="monotone" dataKey="cumulative" name="Cumulative PnL" stroke="#0f766e" dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <div className="empty">Load valid `paper_report.json` data to render PnL charts.</div>
              )}
            </div>
          </article>

          <article className="card">
            <h2>Integrity Status</h2>
            <p>Current health checks and failing gate visibility.</p>

            <div className="check-summary">
              <div className="summary good">
                <div>Checks passing</div>
                <strong>{checkSummary.pass}</strong>
              </div>
              <div className="summary bad">
                <div>Checks failing</div>
                <strong>{checkSummary.fail}</strong>
              </div>
            </div>

            {integrityAlerts.length > 0 ? (
              <div className="alert warning">
                <AlertTriangle className="icon-sm" />
                <div>
                  <strong>Open attention items</strong>
                  <div className="badge-list">
                    {integrityAlerts.map((name) => (
                      <span key={name} className="mini-badge">
                        {name}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="alert success">
                <CheckCircle2 className="icon-sm" />
                <div>
                  <strong>Integrity clean</strong>
                  <div>All monitored checks are currently passing.</div>
                </div>
              </div>
            )}

            <div className="check-list">
              {(health?.checks ?? []).map((check) => (
                <div key={check.name} className="check-row">
                  <span>{check.name}</span>
                  <span className={check.pass ? "mini-badge good" : "mini-badge bad"}>
                    {check.pass ? "PASS" : "FAIL"}
                  </span>
                </div>
              ))}
            </div>
          </article>
        </section>

        <section className="grid-secondary">
          <article className="card">
            <h2>Skip-Reason Distribution</h2>
            <p>Verify that gating behavior stays stable across sessions.</p>
            <div className="chart-box">
              {skipSeries.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={skipSeries}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="value" name="Skips" fill="#ea580c" radius={[9, 9, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="empty">Load valid `paper_report.json` data to render skip reasons.</div>
              )}
            </div>
          </article>

          <article className="card">
            <h2>Integrity Violations by Day</h2>
            <p>Track whether contract or lifecycle errors are drifting over time.</p>
            <div className="chart-box">
              {integritySeries.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={integritySeries}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis allowDecimals={false} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="total" name="Total violations" stroke="#be123c" dot />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="empty">Load valid `paper_report.json` data to render integrity trends.</div>
              )}
            </div>
          </article>
        </section>

        <section className="grid-inputs">
          <article className="card">
            <h2>
              <Upload className="icon-md" /> Load daily_health.json
            </h2>
            <p>Paste or upload the latest health report.</p>
            <input type="file" accept="application/json" onChange={(event) => handleFileLoad(event, setHealthText)} />
            {healthParse.error ? (
              <div className="alert error">
                <AlertTriangle className="icon-sm" />
                <div>Invalid `daily_health.json` format: {healthParse.error}</div>
              </div>
            ) : null}
            <textarea
              value={healthText}
              onChange={(event) => setHealthText(event.target.value)}
              placeholder="Paste daily_health.json here..."
            />
          </article>

          <article className="card">
            <h2>
              <Upload className="icon-md" /> Load paper_report.json
            </h2>
            <p>Paste or upload the latest paper report.</p>
            <input type="file" accept="application/json" onChange={(event) => handleFileLoad(event, setPaperText)} />
            {paperParse.error ? (
              <div className="alert error">
                <AlertTriangle className="icon-sm" />
                <div>Invalid `paper_report.json` format: {paperParse.error}</div>
              </div>
            ) : null}
            <textarea
              value={paperText}
              onChange={(event) => setPaperText(event.target.value)}
              placeholder="Paste paper_report.json here..."
            />
          </article>
        </section>

        <div className="actions">
          <button className="btn primary" onClick={loadDemoData} type="button">
            Load demo data
          </button>
          <button className="btn" onClick={clearInputs} type="button">
            Clear inputs
          </button>
        </div>
      </div>
    </div>
  );
}
