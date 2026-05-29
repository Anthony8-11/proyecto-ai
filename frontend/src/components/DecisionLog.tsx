import type { SimStep } from "../hooks/useSimulation";

const ACTION_LABELS = ["Redirigir", "Activar Gen.", "Monitorear", "Corte"];
const ACTION_COLORS = ["#38bdf8", "#a78bfa", "#22c55e", "#fb923c"];
const COND_COLORS   = ["#22c55e", "#f59e0b", "#ef4444"];

interface Props {
  history: SimStep[];
}

export function DecisionLog({ history }: Props) {
  const recent = [...history].reverse().slice(0, 40);

  return (
    <div className="card">
      <h3>Historial de Decisiones</h3>
      {recent.length === 0 ? (
        <div style={{ color: "#475569", fontSize: ".78rem" }}>Sin datos aún.</div>
      ) : (
        <ul className="log-list">
          {recent.map((s, i) => (
            <li key={i} style={{ display: "flex", alignItems: "center", gap: 6 }}>
              {/* Condition dot */}
              <span style={{
                display: "inline-block", width: 6, height: 6,
                borderRadius: "50%", flexShrink: 0,
                background: COND_COLORS[s.state.env_condition],
              }}/>
              {/* Action tag */}
              <span style={{
                fontSize: ".56rem", fontWeight: 800,
                color: ACTION_COLORS[s.action],
                fontFamily: "monospace",
                minWidth: 44,
              }}>
                {ACTION_LABELS[s.action]}
              </span>
              <span style={{ color: "#2a4060", fontSize: ".6rem" }}>
                S{s.state.position} · {s.state.resources}MW
              </span>
              <span
                className={s.reward >= 0 ? "pos-reward" : "neg-reward"}
                style={{ marginLeft: "auto", fontSize: ".65rem" }}
              >
                {s.reward >= 0 ? "+" : ""}{s.reward}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
