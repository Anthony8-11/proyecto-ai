import type { SimStep } from "../hooks/useSimulation";

const ACTION_LABELS = ["Redirigir flujo", "Activar gen.", "Monitorear", "Corte emerg."];
const ACTION_TAGS   = ["RD", "GEN", "MON", "CRT"];
const ACTION_COLORS = ["#38bdf8", "#a78bfa", "#22c55e", "#fb923c"];
const COND_COLORS   = ["#22c55e", "#f59e0b", "#ef4444"];
const COND_LABELS   = ["Estable", "Sobrecarga", "Fallo de red"];
const SECTOR_NAMES  = ["Industrial", "Res. Norte", "Comercial", "Res. Sur", "Planta"];

interface Props {
  step: SimStep | null;
  connected: boolean;
}

function MiniStat({ label, value, color = "#38bdf8" }: { label: string; value: string | number; color?: string }) {
  return (
    <div className="stat">
      <div className="label">{label}</div>
      <div className="value" style={{ color, fontSize: "1rem" }}>{value}</div>
    </div>
  );
}

export function AgentView({ step, connected }: Props) {
  const cond = step?.state.env_condition ?? 0;
  const condColor = COND_COLORS[cond];

  return (
    <div className="card">
      {/* Header */}
      <div style={{
        display: "flex", alignItems: "center",
        justifyContent: "space-between", marginBottom: ".75rem",
      }}>
        <h3 style={{ margin: 0 }}>Estado del Agente</h3>
        <span className={`conn-badge ${connected ? "ok" : "bad"}`}>
          <span className="conn-dot"/>
          {connected ? "Online" : "Offline"}
        </span>
      </div>

      {/* Condition banner */}
      {step && (
        <div style={{
          display: "flex", alignItems: "center", justifyContent: "space-between",
          background: condColor + "11",
          border: `1px solid ${condColor}33`,
          borderRadius: 8, padding: ".5rem .75rem",
          marginBottom: ".75rem",
          animation: cond > 0 ? "blink-slow 2s infinite" : "none",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
            <span style={{
              display: "inline-block", width: 8, height: 8,
              borderRadius: "50%", background: condColor, flexShrink: 0,
            }}/>
            <span style={{ fontSize: ".75rem", color: condColor, fontWeight: 800 }}>
              Red {COND_LABELS[cond]}
            </span>
          </div>
          <span style={{ fontSize: ".68rem", color: "#475569" }}>
            Sector: <strong style={{ color: condColor }}>S{step.state.position} {SECTOR_NAMES[step.state.position]}</strong>
          </span>
        </div>
      )}

      {/* Stats */}
      <div className="stat-grid" style={{ marginBottom: ".75rem" }}>
        <MiniStat label="Episodio" value={step?.episode ?? "—"} />
        <MiniStat label="Capacidad"
          value={step ? `${step.state.resources} MW` : "—"}
          color={step
            ? step.state.resources > 6 ? "#22c55e"
            : step.state.resources > 3 ? "#f59e0b" : "#ef4444"
            : "#38bdf8"}
        />
        <MiniStat label="Reward paso"
          value={step ? `${step.reward >= 0 ? "+" : ""}${step.reward}` : "—"}
          color={step
            ? step.reward >= 10 ? "#22c55e"
            : step.reward >= 0 ? "#f59e0b" : "#ef4444"
            : "#38bdf8"}
        />
        <MiniStat label="Acumulado"
          value={step?.total_reward ?? "—"}
          color={(step?.total_reward ?? 0) >= 0 ? "#22c55e" : "#ef4444"}
        />
      </div>

      {/* Last action */}
      {step && (
        <div style={{
          display: "flex", alignItems: "center", gap: 8,
          background: "#0b1a2e", borderRadius: 7, padding: ".5rem .75rem",
          border: "1px solid #1e4068", marginBottom: ".75rem",
        }}>
          <span style={{
            fontSize: ".58rem", fontWeight: 900, fontFamily: "monospace",
            color: ACTION_COLORS[step.action],
            background: ACTION_COLORS[step.action] + "18",
            border: `1px solid ${ACTION_COLORS[step.action]}44`,
            borderRadius: 3, padding: "2px 5px",
          }}>
            {ACTION_TAGS[step.action]}
          </span>
          <div>
            <div style={{ fontSize: ".6rem", color: "#475569", textTransform: "uppercase", letterSpacing: ".06em" }}>
              Última acción
            </div>
            <div style={{ fontSize: ".75rem", color: "#38bdf8", fontWeight: 800 }}>
              {ACTION_LABELS[step.action]}
            </div>
          </div>
          {step.q_values && (
            <span style={{ marginLeft: "auto", fontSize: ".65rem", color: "#475569" }}>
              Q={step.q_values[step.action].toFixed(2)}
            </span>
          )}
        </div>
      )}

      {/* Epsilon */}
      <div style={{ marginBottom: ".3rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: ".3rem" }}>
          <span style={{ fontSize: ".62rem", color: "#475569", textTransform: "uppercase", letterSpacing: ".07em" }}>
            Exploración ε
          </span>
          <span style={{ fontSize: ".72rem", color: "#a78bfa", fontWeight: 800 }}>
            {step?.epsilon ?? 1}
          </span>
        </div>
        <div className="epsilon-bar">
          <div className="epsilon-fill" style={{ width: `${(step?.epsilon ?? 1) * 100}%` }}/>
        </div>
      </div>

      {/* ML prediction */}
      {step?.ml_probs && (
        <div style={{
          marginTop: ".6rem",
          background: "#0b1a2e", borderRadius: 7,
          border: "1px solid #162840", padding: ".5rem .7rem",
        }}>
          <div style={{ fontSize: ".6rem", color: "#2a4060", textTransform: "uppercase", letterSpacing: ".08em", marginBottom: ".35rem" }}>
            ML — Predicción próximo estado
          </div>
          <div style={{ display: "flex", gap: ".5rem" }}>
            {[
              { label: "Estable",   color: "#22c55e", p: step.ml_probs[0] ?? 0 },
              { label: "Sobrecar.", color: "#f59e0b", p: step.ml_probs[1] ?? 0 },
              { label: "Fallo",     color: "#ef4444", p: step.ml_probs[2] ?? 0 },
            ].map(({ label, color, p }) => (
              <div key={label} style={{ flex: 1, display: "flex", flexDirection: "column", gap: 2 }}>
                <div style={{
                  height: Math.max(3, Math.round(p * 32)),
                  background: color, borderRadius: "2px 2px 0 0",
                  opacity: step.ml_trained ? 1 : 0.3,
                  transition: "height .4s cubic-bezier(.4,0,.2,1)",
                  alignSelf: "flex-end", width: "100%",
                  boxShadow: `0 0 6px ${color}55`,
                }}/>
                <span style={{ fontSize: ".58rem", color, fontWeight: 700, textAlign: "center" }}>
                  {(p * 100).toFixed(0)}%
                </span>
                <span style={{ fontSize: ".55rem", color: "#475569", textAlign: "center" }}>
                  {label}
                </span>
              </div>
            ))}
          </div>
          {!step.ml_trained && (
            <div style={{ fontSize: ".6rem", color: "#475569", textAlign: "center", marginTop: ".25rem" }}>
              Sin entrenar aún (&lt;50 pasos)
            </div>
          )}
        </div>
      )}
    </div>
  );
}
