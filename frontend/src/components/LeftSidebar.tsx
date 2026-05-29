import type { ActiveView, SimStats, SimStep } from "../hooks/useSimulation";

const SECTORS = [
  { label: "Industrial",  short: "S0", abbr: "IND" },
  { label: "Res. Norte",  short: "S1", abbr: "RES" },
  { label: "Comercial",   short: "S2", abbr: "COM" },
  { label: "Res. Sur",    short: "S3", abbr: "SUR" },
  { label: "Planta",      short: "S4", abbr: "PLT" },
];

const COND_COLOR = ["#22c55e", "#f59e0b", "#ef4444"];
const COND_LABEL = ["Estable", "Sobrecarga", "Fallo de red"];

const NAV_ITEMS: { view: ActiveView; label: string }[] = [
  { view: "operaciones",   label: "Operaciones"  },
  { view: "entrenamiento", label: "Entrenamiento" },
  { view: "reglas",        label: "Reglas Prolog" },
  { view: "comparacion",   label: "Comparación"   },
  { view: "historico",     label: "Histórico"     },
];

interface Props {
  step: SimStep | null;
  stats: SimStats | null;
  activeView: ActiveView;
  onNavigate: (v: ActiveView) => void;
  connected: boolean;
}

export function LeftSidebar({ step, stats, activeView, onNavigate, connected }: Props) {
  const cond = step?.state.env_condition ?? 0;
  const pos  = step?.state.position ?? -1;

  return (
    <aside className="left-sidebar">

      {/* ── Brand ──────────────────────────────────────────── */}
      <div style={{
        padding: ".85rem .9rem .7rem",
        borderBottom: "1px solid var(--border)",
      }}>
        <div style={{
          fontSize: ".82rem", fontWeight: 900,
          color: "var(--blue)", letterSpacing: ".06em",
        }}>
          SMART·GRID
        </div>
        <div style={{ fontSize: ".52rem", color: "var(--text-4)", marginTop: 2 }}>
          v0.1.0 · RL Dashboard
        </div>
      </div>

      {/* ── Navigation ─────────────────────────────────────── */}
      <div style={{ padding: ".5rem 0" }}>
        <div className="sidebar-section-label">Navegación</div>
        {NAV_ITEMS.map(({ view, label }) => (
          <button
            key={view}
            className={`nav-item ${activeView === view ? "active" : ""}`}
            onClick={() => onNavigate(view)}
            type="button"
          >
            {label}
          </button>
        ))}
      </div>

      {/* ── Sector status ──────────────────────────────────── */}
      <div style={{ padding: ".4rem 0", borderTop: "1px solid var(--border)" }}>
        <div className="sidebar-section-label">Red — 5 sectores</div>
        {SECTORS.map((s, i) => {
          const isActive = i === pos;
          const cap = isActive && step ? step.state.resources : null;
          const capPct = cap !== null ? (cap / 10) * 100 : null;
          const capColor = cap !== null
            ? cap > 6 ? "#22c55e" : cap > 3 ? "#f59e0b" : "#ef4444"
            : "#2a4060";

          return (
            <div key={i} style={{
              display: "flex", alignItems: "center", gap: 7,
              padding: ".28rem .85rem",
              borderLeft: isActive ? `2px solid ${COND_COLOR[cond]}` : "2px solid transparent",
              background: isActive ? `${COND_COLOR[cond]}09` : "transparent",
              transition: "all .2s",
            }}>
              {/* Sector abbr badge */}
              <span style={{
                fontSize: ".5rem", fontWeight: 800, fontFamily: "monospace",
                background: isActive ? `${COND_COLOR[cond]}22` : "var(--bg)",
                color: isActive ? COND_COLOR[cond] : "var(--text-4)",
                border: `1px solid ${isActive ? COND_COLOR[cond] + "55" : "var(--border)"}`,
                borderRadius: 3, padding: "1px 4px", minWidth: 22, textAlign: "center",
              }}>
                {s.abbr}
              </span>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{
                  fontSize: ".6rem", fontWeight: isActive ? 800 : 500,
                  color: isActive ? COND_COLOR[cond] : "var(--text-3)",
                  whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
                }}>
                  {s.short} {s.label}
                </div>
                {capPct !== null && (
                  <div style={{
                    height: 2, background: "var(--border)", borderRadius: 1,
                    marginTop: 2, overflow: "hidden",
                  }}>
                    <div style={{
                      height: "100%", width: `${capPct}%`,
                      background: capColor, borderRadius: 1,
                      transition: "width .4s",
                    }}/>
                  </div>
                )}
              </div>
              {cap !== null && (
                <span style={{ fontSize: ".58rem", color: capColor, fontWeight: 800 }}>
                  {cap}MW
                </span>
              )}
            </div>
          );
        })}
      </div>

      {/* ── Dinámica activa ────────────────────────────────── */}
      <div style={{ padding: ".4rem 0", borderTop: "1px solid var(--border)" }}>
        <div className="sidebar-section-label">Dinámica activa</div>
        <div style={{ padding: ".2rem .85rem", display: "flex", flexDirection: "column", gap: 4 }}>
          {step ? (
            <div style={{
              display: "flex", alignItems: "center", gap: 7,
              background: `${COND_COLOR[cond]}11`,
              border: `1px solid ${COND_COLOR[cond]}33`,
              borderRadius: 5, padding: "4px 8px",
              animation: cond > 0 ? "blink-slow 2s infinite" : "none",
            }}>
              <span style={{
                display: "inline-block", width: 6, height: 6,
                borderRadius: "50%", background: COND_COLOR[cond], flexShrink: 0,
              }}/>
              <span style={{ fontSize: ".62rem", color: COND_COLOR[cond], fontWeight: 700 }}>
                {COND_LABEL[cond]}
              </span>
            </div>
          ) : (
            <span style={{ fontSize: ".6rem", color: "var(--text-4)" }}>Sin simulación</span>
          )}
        </div>
      </div>

      {/* ── Estadísticas ───────────────────────────────────── */}
      {stats && (
        <div style={{ padding: ".4rem 0", borderTop: "1px solid var(--border)" }}>
          <div className="sidebar-section-label">Estadísticas</div>
          <div style={{ padding: "0 .85rem", display: "flex", flexDirection: "column", gap: 5 }}>
            {[
              {
                label: "Pasos",
                value: stats.steps,
                color: "var(--blue)",
              },
              {
                label: "Episodios",
                value: stats.episodes,
                color: "var(--text-2)",
              },
              {
                label: "Media ult. 10",
                value: stats.mean_reward_last10 ?? 0,
                color: (stats.mean_reward_last10 ?? 0) >= 0 ? "#22c55e" : "#ef4444",
              },
              {
                label: "Cap. promedio",
                value: stats.avg_resources != null ? `${stats.avg_resources} MW` : "—",
                color: "#a78bfa",
              },
            ].map(({ label, value, color }) => (
              <div key={label} style={{
                display: "flex", justifyContent: "space-between", alignItems: "center",
              }}>
                <span style={{
                  fontSize: ".58rem", color: "var(--text-4)",
                  textTransform: "uppercase", letterSpacing: ".08em",
                }}>
                  {label}
                </span>
                <span style={{ fontSize: ".68rem", color, fontWeight: 800, fontVariantNumeric: "tabular-nums" }}>
                  {String(value)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Sistema ────────────────────────────────────────── */}
      <div style={{
        padding: ".4rem 0",
        borderTop: "1px solid var(--border)",
        marginTop: "auto",
      }}>
        <div className="sidebar-section-label">Sistema</div>
        <div style={{ padding: "0 .85rem", display: "flex", flexDirection: "column", gap: 3 }}>
          {[
            { label: "Política", value: step?.agent_type === "dqn" ? "NEURO_DQN" : "Q-TABLE",   color: "#38bdf8" },
            { label: "Episodio", value: step?.episode ?? "—",                                    color: "var(--text-2)" },
            { label: "Prolog",   value: step?.logic_engine === "prolog" ? "OK" : step?.logic_engine ? "PY" : "—",
              color: step?.logic_engine === "prolog" ? "#22c55e" : "#f59e0b" },
            { label: "ML",       value: step?.ml_trained ? "OK" : "—",
              color: step?.ml_trained ? "#22c55e" : "var(--text-4)" },
            { label: "WS",       value: connected ? "Conectado" : "Sin conexión",
              color: connected ? "#22c55e" : "#ef4444" },
          ].map(({ label, value, color }) => (
            <div key={label} style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: ".58rem", color: "var(--text-4)", textTransform: "uppercase", letterSpacing: ".08em" }}>
                {label}
              </span>
              <span style={{ fontSize: ".62rem", color, fontWeight: 700 }}>
                {String(value)}
              </span>
            </div>
          ))}
        </div>
      </div>

    </aside>
  );
}
