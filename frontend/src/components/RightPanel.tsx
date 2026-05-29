import { useEffect, useMemo, useState } from "react";
import type { AgentType, SimStats, SimStep } from "../hooks/useSimulation";

type SaveStatus = "idle" | "saving" | "ok" | "error";

/* ── SVG Sparkline ──────────────────────────────────────────────── */
function Sparkline({
  values, color, width = 238, height = 38,
}: {
  values: number[]; color: string; width?: number; height?: number;
}) {
  if (values.length < 2) {
    // Visible "waiting for data" placeholder — grid + dashed center line
    return (
      <svg width={width} height={height} style={{ display: "block" }}>
        {/* subtle grid */}
        <line x1={0} y1={height * 0.25} x2={width} y2={height * 0.25}
          stroke={color} strokeWidth={0.5} strokeOpacity={0.08}/>
        <line x1={0} y1={height * 0.5}  x2={width} y2={height * 0.5}
          stroke={color} strokeWidth={0.5} strokeOpacity={0.08}/>
        <line x1={0} y1={height * 0.75} x2={width} y2={height * 0.75}
          stroke={color} strokeWidth={0.5} strokeOpacity={0.08}/>
        {/* dashed center line */}
        <line x1={0} y1={height / 2} x2={width} y2={height / 2}
          stroke={color} strokeWidth={1.5} strokeOpacity={0.35} strokeDasharray="4 4"/>
        {/* waiting dots */}
        {[0.2, 0.5, 0.8].map(p => (
          <circle key={p} cx={p * width} cy={height / 2} r={2}
            fill={color} fillOpacity={0.25}/>
        ))}
      </svg>
    );
  }

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  const toY = (v: number) =>
    height - ((v - min) / range) * (height - 8) - 4;

  const pts = values.map((v, i) => {
    const x = (i / (values.length - 1)) * width;
    return `${x.toFixed(1)},${toY(v).toFixed(1)}`;
  }).join(" ");

  const last  = values[values.length - 1];
  const lastX = width;
  const lastY = toY(last);

  // Filled area under the line
  const areaPath = [
    `M 0,${toY(values[0]).toFixed(1)}`,
    ...values.map((v, i) => {
      const x = (i / (values.length - 1)) * width;
      return `L ${x.toFixed(1)},${toY(v).toFixed(1)}`;
    }),
    `L ${width},${height}`,
    `L 0,${height}`,
    "Z",
  ].join(" ");

  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      {/* filled area */}
      <path d={areaPath} fill={color} fillOpacity={0.08}/>
      {/* line */}
      <polyline
        points={pts}
        fill="none"
        stroke={color}
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeOpacity={0.9}
      />
      {/* current-value dot */}
      <circle cx={lastX} cy={lastY} r={3} fill={color} fillOpacity={0.9}/>
      <circle cx={lastX} cy={lastY} r={5} fill={color} fillOpacity={0.2}/>
    </svg>
  );
}

/* ── Metric card ────────────────────────────────────────────────── */
function MetricCard({
  label, value, unit = "", color, sub,
}: {
  label: string; value: string | number; unit?: string; color: string; sub?: string;
}) {
  return (
    <div className="metric-card">
      <div style={{ fontSize: ".52rem", color: "var(--text-4)", textTransform: "uppercase", letterSpacing: ".1em", marginBottom: ".2rem" }}>
        {label}
      </div>
      <div style={{ fontSize: "1.15rem", fontWeight: 900, color, lineHeight: 1, animation: "number-pop .25s ease" }}>
        {value}<span style={{ fontSize: ".6rem", fontWeight: 500, marginLeft: 2 }}>{unit}</span>
      </div>
      {sub && (
        <div style={{ fontSize: ".52rem", color: "var(--text-4)", marginTop: 2 }}>{sub}</div>
      )}
    </div>
  );
}

/* ── Sparkline card ─────────────────────────────────────────────── */
function SparkCard({
  label, values, color, lastLabel,
}: {
  label: string; values: number[]; color: string; lastLabel?: string;
}) {
  const last = values.length > 0 ? values[values.length - 1] : null;
  return (
    <div className="sparkline-card">
      <div style={{
        display: "flex", justifyContent: "space-between", alignItems: "center",
        marginBottom: ".3rem",
      }}>
        <span style={{ fontSize: ".58rem", color: "var(--text-3)", fontWeight: 700, textTransform: "uppercase", letterSpacing: ".07em" }}>
          {label}
        </span>
        {last !== null && (
          <span style={{ fontSize: ".65rem", color, fontWeight: 800 }}>
            {typeof last === "number" ? last.toFixed(1) : last}{lastLabel}
          </span>
        )}
      </div>
      <Sparkline values={values} color={color}/>
    </div>
  );
}

/* ── Main component ─────────────────────────────────────────────── */
interface Props {
  step: SimStep | null;
  history: SimStep[];
  stats: SimStats | null;
  running: boolean;
  connected: boolean;
  agentType: AgentType;
  onStart: () => void;
  onPause: () => void;
  onReset: () => void;
  onSwitchAgent: (t: AgentType) => void;
  onPauseAndSave: () => void;
  onSetAutoSaveInterval: (n: number) => void;
  saveStatus?: SaveStatus;
}

export function RightPanel({
  step, history, stats, running, connected,
  agentType, onStart, onPause, onReset, onSwitchAgent,
  onPauseAndSave, onSetAutoSaveInterval, saveStatus = "idle",
}: Props) {

  const epsilon = step?.epsilon ?? 1.0;

  // Local state for the auto-save interval selector.
  // Initialized from stats and kept in sync when stats change from the server,
  // but updated immediately on user interaction (optimistic update) so the
  // select never snaps back to the old value while the API call is in flight.
  const [autoSaveEvery, setAutoSaveEvery] = useState<number>(
    stats?.auto_save_every ?? 1,
  );
  useEffect(() => {
    if (stats?.auto_save_every != null) {
      setAutoSaveEvery(stats.auto_save_every);
    }
  }, [stats?.auto_save_every]);

  // ── Derived metrics — memoized so they only recompute when history changes
  const { tasaExito, rewardAvg, violaciones } = useMemo(() => {
    if (history.length === 0) return { tasaExito: 0, rewardAvg: 0, violaciones: 0 };

    // Single pass over the last 100 entries for the three scalar metrics
    const start100 = Math.max(0, history.length - 100);
    const start20  = Math.max(0, history.length - 20);
    let successCount = 0;
    let rewardSum    = 0;
    let violCount    = 0;
    for (let i = start100; i < history.length; i++) {
      const s = history[i];
      if (s.reward >= 10) successCount++;
      if (s.allowed_actions.length < 4) violCount++;
      if (i >= start20) rewardSum += s.reward;
    }
    const n100 = history.length - start100;
    const n20  = history.length - start20;
    return {
      tasaExito:   Math.round((successCount / n100) * 100),
      rewardAvg:   n20 > 0 ? Math.round(rewardSum / n20) : 0,
      violaciones: violCount,
    };
  }, [history]);

  // ── Sparkline data series — memoized separately (different window size)
  const WINDOW = 60;
  const { rewardSeries, successSeries, violSeries, capSeries } = useMemo(() => {
    const recent = history.slice(-WINDOW);
    if (recent.length === 0) {
      return { rewardSeries: [], successSeries: [], violSeries: [], capSeries: [] };
    }

    // Use individual step reward (+10 / -5 / -10) — always has variation.
    // total_reward resets each episode and can appear flat when many steps
    // start at 0 within the same window.
    const rewardS:  number[] = [];
    const successS: number[] = [];
    const violS:    number[] = [];
    const capS:     number[] = [];

    for (let i = 0; i < recent.length; i++) {
      const s = recent[i];
      rewardS.push(s.reward);           // +10 / -5 / -10 per step
      capS.push(s.state.resources);     // 0-10 MW

      // Rolling 10-step window for rate metrics
      const wStart = Math.max(0, i - 9);
      let sc = 0, vc = 0;
      for (let j = wStart; j <= i; j++) {
        if (recent[j].reward >= 10)                  sc++;
        if (recent[j].allowed_actions.length < 4)    vc++;
      }
      const wLen = i - wStart + 1;
      successS.push(Math.round((sc / wLen) * 100));
      violS.push(vc);
    }

    return { rewardSeries: rewardS, successSeries: successS, violSeries: violS, capSeries: capS };
  }, [history]);

  return (
    <aside className="right-panel">

      {/* ── Key metrics 2×2 ─────────────────────────────── */}
      <div className="sidebar-section-label" style={{ paddingLeft: 0, paddingTop: 0 }}>
        Métricas clave
      </div>
      <div className="metric-cards-grid">
        <MetricCard
          label="Tasa éxito"
          value={tasaExito}
          unit="%"
          color={tasaExito >= 60 ? "#22c55e" : tasaExito >= 30 ? "#f59e0b" : "#ef4444"}
          sub="últimos 100 pasos"
        />
        <MetricCard
          label="Reward avg"
          value={rewardAvg >= 0 ? `+${rewardAvg}` : String(rewardAvg)}
          color={rewardAvg >= 5 ? "#22c55e" : rewardAvg >= 0 ? "#f59e0b" : "#ef4444"}
          sub="últimos 20 pasos"
        />
        <MetricCard
          label="Violaciones"
          value={violaciones}
          color={violaciones > 20 ? "#ef4444" : violaciones > 5 ? "#f59e0b" : "#22c55e"}
          sub="Prolog activo"
        />
        <MetricCard
          label="Epsilon ε"
          value={epsilon.toFixed(3)}
          color="#a78bfa"
          sub={epsilon > 0.5 ? "explorando" : "explotando"}
        />
      </div>

      {/* ── Epsilon bar ─────────────────────────────────── */}
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: ".25rem" }}>
          <span style={{ fontSize: ".55rem", color: "var(--text-4)", textTransform: "uppercase", letterSpacing: ".08em" }}>
            Exploración ε
          </span>
          <span style={{ fontSize: ".6rem", color: "#a78bfa", fontWeight: 800 }}>
            {(epsilon * 100).toFixed(1)}%
          </span>
        </div>
        <div className="epsilon-bar">
          <div className="epsilon-fill" style={{ width: `${epsilon * 100}%` }}/>
        </div>
      </div>

      {/* ── Live sparklines ─────────────────────────────── */}
      <div>
        <div style={{
          display: "flex", justifyContent: "space-between", alignItems: "center",
          marginBottom: ".55rem",
        }}>
          <span style={{
            fontSize: ".52rem", textTransform: "uppercase", letterSpacing: ".12em",
            color: "var(--text-4)", fontWeight: 700,
          }}>
            Entrenamiento en vivo
          </span>
          <span style={{
            fontSize: ".5rem", padding: "1px 6px", borderRadius: 3, fontWeight: 700,
            background: history.length > 0 ? "#052e16" : "#0a1628",
            color:      history.length > 0 ? "#4ade80" : "#2a4060",
            border:     `1px solid ${history.length > 0 ? "#166534" : "#1e293b"}`,
          }}>
            {history.length > 0 ? `${history.length} pasos` : "Sin datos"}
          </span>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: ".45rem" }}>
          <SparkCard label="Reward / paso" values={rewardSeries} color="#22c55e" lastLabel=""/>
          <SparkCard label="Tasa éxito %" values={successSeries} color="#38bdf8" lastLabel="%"/>
          <SparkCard label="Actividad simbólica" values={violSeries} color="#f59e0b" lastLabel=""/>
          <SparkCard label="Capacidad MW" values={capSeries} color="#a78bfa" lastLabel=" MW"/>
        </div>
      </div>

      {/* ── DQN telemetry ───────────────────────────────── */}
      {agentType === "dqn" && step?.dqn_loss !== undefined && (
        <div>
          <div className="sidebar-section-label" style={{ paddingLeft: 0 }}>DQN Telemetría</div>
          <div className="metric-cards-grid">
            <MetricCard label="Loss MSE" value={step.dqn_loss.toFixed(4)} color="#f59e0b"/>
            <MetricCard label="Buffer" value={step.dqn_buffer ?? 0} color="#a78bfa"/>
          </div>
        </div>
      )}

      {/* ── Agent selector ──────────────────────────────── */}
      <div>
        <div className="sidebar-section-label" style={{ paddingLeft: 0 }}>Agente RL</div>
        <div className="btn-row" style={{ marginBottom: ".4rem" }}>
          {(["qlearning", "dqn"] as AgentType[]).map(t => (
            <button
              key={t}
              className={agentType === t ? "primary" : "secondary"}
              style={{ flex: 1, fontSize: ".68rem" }}
              onClick={() => onSwitchAgent(t)}
              disabled={running}
              type="button"
            >
              {t === "qlearning" ? "Q-Learn" : "DQN"}
            </button>
          ))}
        </div>
        <p style={{ fontSize: ".58rem", color: "var(--text-4)", margin: 0 }}>
          {agentType === "qlearning"
            ? "Q-tabla 165×4 · ε-greedy · Bellman"
            : "MLP 3→64→64→4 · replay · target net"}
        </p>
      </div>

      {/* ── Auto-guardado ───────────────────────────────── */}
      <div>
        <div className="sidebar-section-label" style={{ paddingLeft: 0 }}>Auto-guardado</div>

        {/* Last-save indicator */}
        <div style={{
          display: "flex", justifyContent: "space-between", alignItems: "center",
          marginBottom: ".45rem",
          background: "#0a1628", border: "1px solid #1e293b",
          borderRadius: 6, padding: ".4rem .65rem",
        }}>
          <span style={{ fontSize: ".6rem", color: "#475569" }}>Último guardado</span>
          <span style={{ fontSize: ".68rem", fontWeight: 700, color: "#22c55e" }}>
            {stats?.last_saved_episode != null && stats.last_saved_episode > 0
              ? `ep. ${stats.last_saved_episode}`
              : "—"}
          </span>
        </div>

        {/* Unsaved episodes warning */}
        {stats && stats.last_saved_episode != null && (
          (() => {
            const unsaved = stats.episodes - 1 - stats.last_saved_episode;
            if (unsaved <= 0) return null;
            const warn = unsaved >= 5;
            return (
              <div style={{
                fontSize: ".6rem", marginBottom: ".45rem",
                padding: ".3rem .65rem", borderRadius: 5,
                background: warn ? "#2d1a0a" : "#0a1628",
                border: `1px solid ${warn ? "#92400e" : "#1e293b"}`,
                color: warn ? "#f59e0b" : "#475569",
                display: "flex", justifyContent: "space-between",
              }}>
                <span>Sin guardar</span>
                <span style={{ fontWeight: 700 }}>{unsaved} ep.</span>
              </div>
            );
          })()
        )}

        {/* Interval selector */}
        <div style={{ display: "flex", alignItems: "center", gap: ".5rem", marginBottom: ".45rem" }}>
          <span style={{ fontSize: ".6rem", color: "#475569", whiteSpace: "nowrap" }}>
            Guardar cada
          </span>
          <select
            value={autoSaveEvery}
            onChange={e => {
              const n = Number(e.target.value);
              setAutoSaveEvery(n);          // update immediately — no flicker
              onSetAutoSaveInterval(n);     // sync to backend in background
            }}
            style={{
              flex: 1, fontSize: ".65rem", background: "#0f172a",
              color: "#e2e8f0", border: "1px solid #334155", borderRadius: 4,
              padding: "2px 4px", cursor: "pointer",
            }}
          >
            {[1, 2, 3, 5, 10].map(n => (
              <option key={n} value={n}>{n} ep.</option>
            ))}
          </select>
        </div>

        {/* Pause-and-save button — safe exit for long sessions */}
        <button
          className="secondary"
          onClick={onPauseAndSave}
          type="button"
          disabled={saveStatus === "saving"}
          style={{
            width: "100%", fontSize: ".65rem", marginBottom: ".35rem",
            color: saveStatus === "ok" ? "#4ade80" : saveStatus === "error" ? "#f87171"
              : running ? "#f59e0b" : undefined,
            borderColor: saveStatus === "ok" ? "#166534" : saveStatus === "error" ? "#7f1d1d"
              : running ? "#92400e" : undefined,
            transition: "color .2s, border-color .2s",
          }}
        >
          {saveStatus === "saving" ? "Guardando…"
            : saveStatus === "ok" ? "Guardado ✓"
            : saveStatus === "error" ? "Error al guardar"
            : running ? "Pausar y guardar"
            : "Guardar estado"}
        </button>
      </div>

      {/* ── Controls ────────────────────────────────────── */}
      <div>
        <div className="sidebar-section-label" style={{ paddingLeft: 0 }}>Control</div>
        <div className="btn-row">
          <button className="primary" onClick={onStart} disabled={running || !connected} type="button" style={{ flex: 1, fontSize: ".7rem" }}>
            ▶
          </button>
          <button className="secondary" onClick={onPause} disabled={!running} type="button" style={{ flex: 1, fontSize: ".7rem" }}>
            ⏸
          </button>
          <button className="danger" onClick={onReset} type="button" style={{ flex: 1, fontSize: ".7rem" }}>
            ↺
          </button>
        </div>
      </div>


    </aside>
  );
}
