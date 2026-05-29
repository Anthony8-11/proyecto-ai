import type { SimStep } from "../hooks/useSimulation";

interface Props { step: SimStep | null; }

const ACTION_LABELS = ["Redirigir flujo", "Activar generadores", "Monitorear", "Corte emergencia"];
const ACTION_TAGS   = ["RD", "GEN", "MON", "CRT"];
const ACTION_COLORS = ["#38bdf8", "#a78bfa", "#22c55e", "#fb923c"];
const COND_LABELS   = ["Estable", "Sobrecarga", "Fallo de red"];
const COND_COLORS   = ["#22c55e", "#f59e0b", "#ef4444"];

/* ── Mini step label ──────────────────────────────────── */
function PipeStep({ n, label, accent = "#38bdf8" }: { n: number; label: string; accent?: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: ".4rem" }}>
      <div style={{
        width: 20, height: 20, borderRadius: "50%",
        background: accent + "22", border: `1.5px solid ${accent}`,
        color: accent, fontSize: ".6rem", fontWeight: 900,
        display: "flex", alignItems: "center", justifyContent: "center",
        flexShrink: 0,
      }}>
        {n}
      </div>
      <span style={{
        fontSize: ".6rem", color: "#2a4060",
        fontWeight: 700, textTransform: "uppercase", letterSpacing: ".08em",
      }}>
        {label}
      </span>
    </div>
  );
}

/* ── Connector arrow ─────────────────────────────────── */
function Arrow() {
  return (
    <div style={{
      display: "flex", alignItems: "center", justifyContent: "center",
      color: "#1e4068", fontSize: ".9rem", flexShrink: 0,
    }}>
      ›
    </div>
  );
}

/* ── Individual pipeline card ────────────────────────── */
function PipeCard({
  n, label, accent, children, minWidth = 160,
}: {
  n: number; label: string; accent?: string;
  children: React.ReactNode; minWidth?: number;
}) {
  return (
    <div style={{
      flex: "0 0 auto", minWidth, maxWidth: 240,
      background: "#07111f",
      border: "1px solid #162840",
      borderRadius: 10, padding: ".75rem",
      display: "flex", flexDirection: "column",
      animation: "fade-up .3s ease",
    }}>
      <PipeStep n={n} label={label} accent={accent}/>
      {children}
    </div>
  );
}

/* ── Chip badges ─────────────────────────────────────── */
function Chip({ label, color, bg }: { label: string; color: string; bg: string }) {
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 4,
      background: bg, color, border: `1px solid ${color}44`,
      borderRadius: 5, padding: "2px 7px",
      fontSize: ".68rem", fontWeight: 700, whiteSpace: "nowrap",
    }}>
      {label}
    </span>
  );
}

/* ── Prob mini bar ───────────────────────────────────── */
function ProbBar({ label, prob, color }: { label: string; prob: number; color: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <span style={{ fontSize: ".62rem", color: "#475569", minWidth: 52 }}>{label}</span>
      <div style={{
        flex: 1, height: 6, borderRadius: 3,
        background: "#0b1a2e", overflow: "hidden", border: "1px solid #162840",
      }}>
        <div style={{
          width: `${prob * 100}%`, height: "100%",
          background: color, borderRadius: 3,
          transition: "width .4s cubic-bezier(.4,0,.2,1)",
          boxShadow: `0 0 4px ${color}66`,
        }}/>
      </div>
      <span style={{ fontSize: ".62rem", color, fontWeight: 700, minWidth: 28 }}>
        {(prob * 100).toFixed(0)}%
      </span>
    </div>
  );
}

export function DecisionAnatomy({ step }: Props) {
  if (!step) {
    return (
      <div className="card">
        <h3>Motor de Decisión — Pipeline RL</h3>
        <p style={{ fontSize: ".78rem", color: "#2a4060", marginTop: ".25rem" }}>
          Inicia la simulación para ver el flujo de decisión en tiempo real.
        </p>
      </div>
    );
  }

  const {
    state, action, allowed_actions, reward, epsilon,
    ml_probs, ml_trained, q_values, decision_mode, agent_type,
  } = step;

  const condColor = COND_COLORS[state.env_condition];
  const condLabel = COND_LABELS[state.env_condition];
  const rewardColor = reward >= 10 ? "#22c55e" : reward >= 0 ? "#f59e0b" : "#ef4444";
  const rewardLabel = reward >= 10 ? "Correcto" : reward >= 0 ? "Neutro" : "Penalización";
  const isExplore = decision_mode === "exploración";

  return (
    <div className="card" style={{ padding: "1rem 1.25rem" }}>
      <div style={{
        display: "flex", alignItems: "center",
        justifyContent: "space-between", marginBottom: ".85rem",
      }}>
        <span style={{
          fontSize: ".65rem", fontWeight: 800, textTransform: "uppercase",
          letterSpacing: ".1em", color: "#2a4060",
        }}>
          Pipeline de Decisión
        </span>
        <div style={{ display: "flex", gap: ".4rem" }}>
          <span style={{
            fontSize: ".62rem", padding: "1px 7px", borderRadius: 4,
            background: "#0b1a2e", border: "1px solid #1e4068", color: "#38bdf8", fontWeight: 700,
          }}>
            {agent_type === "dqn" ? "DQN" : "Q-Learning"}
          </span>
          <span style={{
            fontSize: ".62rem", padding: "1px 7px", borderRadius: 4,
            background: isExplore ? "#1e1b4b" : "#052e16",
            border: `1px solid ${isExplore ? "#6d28d9" : "#166534"}`,
            color: isExplore ? "#a78bfa" : "#4ade80", fontWeight: 700,
          }}>
            {isExplore ? "Exploración" : "Explotación"}
          </span>
        </div>
      </div>

      {/* Horizontal pipeline */}
      <div style={{
        display: "flex", gap: ".5rem", alignItems: "stretch",
        overflowX: "auto", paddingBottom: ".25rem",
      }}>

        {/* Step 1 — Estado */}
        <PipeCard n={1} label="Estado" accent="#38bdf8">
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {[
              { k: "Sector",    v: `S${state.position}`, c: "#38bdf8" },
              { k: "Capacidad", v: `${state.resources} MW`, c: state.resources > 6 ? "#22c55e" : state.resources > 3 ? "#f59e0b" : "#f87171" },
              { k: "Red",       v: condLabel, c: condColor },
            ].map(({ k, v, c }) => (
              <div key={k} style={{
                display: "flex", justifyContent: "space-between",
                background: "#0b1a2e", borderRadius: 5, padding: "3px 7px",
                border: "1px solid #162840",
              }}>
                <span style={{ fontSize: ".62rem", color: "#475569" }}>{k}</span>
                <span style={{ fontSize: ".65rem", color: c, fontWeight: 800 }}>{v}</span>
              </div>
            ))}
          </div>
        </PipeCard>

        <Arrow/>

        {/* Step 2 — Prolog */}
        <PipeCard n={2} label="Prolog — Reglas" accent="#06b6d4" minWidth={150}>
          <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
            {[0, 1, 2, 3].map(a => {
              const ok = allowed_actions.includes(a);
              return (
                <div key={a} style={{
                  display: "flex", alignItems: "center", gap: 5,
                  background: ok ? "#052e1644" : "#2d050544",
                  border: `1px solid ${ok ? "#166534" : "#7f1d1d"}`,
                  borderRadius: 4, padding: "2px 6px",
                }}>
                  <span style={{ fontSize: ".65rem", fontWeight: 700, color: ok ? "#4ade80" : "#f87171" }}>
                    {ok ? "✓" : "✗"}
                  </span>
                  <span style={{
                    fontSize: ".52rem", fontWeight: 800, fontFamily: "monospace",
                    color: ACTION_COLORS[a],
                    background: ACTION_COLORS[a] + "18",
                    border: `1px solid ${ACTION_COLORS[a]}44`,
                    borderRadius: 3, padding: "1px 4px",
                  }}>
                    {ACTION_TAGS[a]}
                  </span>
                  <span style={{ fontSize: ".6rem", color: ok ? "#4ade80" : "#ef4444" }}>
                    {ACTION_LABELS[a].split(" ")[0]}
                  </span>
                </div>
              );
            })}
          </div>
        </PipeCard>

        <Arrow/>

        {/* Step 3 — ML */}
        <PipeCard n={3} label="ML — RandomForest" accent="#a78bfa" minWidth={150}>
          {!ml_trained && (
            <span style={{ fontSize: ".62rem", color: "#475569", marginBottom: 4 }}>
              Sin entrenar aún
            </span>
          )}
          <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
            <ProbBar label="Estable"    prob={ml_probs[0] ?? 0} color="#22c55e"/>
            <ProbBar label="Sobrecar."  prob={ml_probs[1] ?? 0} color="#f59e0b"/>
            <ProbBar label="Fallo"      prob={ml_probs[2] ?? 0} color="#ef4444"/>
          </div>
        </PipeCard>

        <Arrow/>

        {/* Step 4 — RL */}
        <PipeCard n={4} label={agent_type === "dqn" ? "DQN — Red Neuronal" : "Q-Learning — Tabla"} accent="#fb923c" minWidth={165}>
          <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
            <div style={{
              display: "flex", alignItems: "center", gap: 6,
              background: "#0c1829", borderRadius: 6, padding: "5px 8px",
              border: "1px solid #0ea5e9",
            }}>
              <span style={{
                fontSize: ".55rem", fontWeight: 900, fontFamily: "monospace",
                color: ACTION_COLORS[action],
                background: ACTION_COLORS[action] + "18",
                border: `1px solid ${ACTION_COLORS[action]}44`,
                borderRadius: 3, padding: "2px 5px",
              }}>
                {ACTION_TAGS[action]}
              </span>
              <div>
                <div style={{ fontSize: ".72rem", color: "#38bdf8", fontWeight: 800 }}>
                  {ACTION_LABELS[action]}
                </div>
                {q_values && (
                  <div style={{ fontSize: ".6rem", color: "#475569" }}>
                    Q = {q_values[action].toFixed(2)}
                  </div>
                )}
              </div>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
              <span style={{ fontSize: ".6rem", color: "#475569" }}>ε =</span>
              <span style={{ fontSize: ".72rem", color: "#a78bfa", fontWeight: 800 }}>{epsilon}</span>
              <div style={{ flex: 1, height: 4, borderRadius: 2, background: "#0b1a2e", overflow: "hidden" }}>
                <div style={{ width: `${Number(epsilon) * 100}%`, height: "100%", background: "#a78bfa", borderRadius: 2 }}/>
              </div>
            </div>
          </div>
        </PipeCard>

        <Arrow/>

        {/* Step 5 — Reward */}
        <PipeCard n={5} label="Resultado" accent={rewardColor} minWidth={110}>
          <div style={{
            display: "flex", flexDirection: "column", alignItems: "center",
            justifyContent: "center", flex: 1, gap: 6,
          }}>
            <div style={{
              fontSize: "1.8rem", fontWeight: 900,
              color: rewardColor, lineHeight: 1,
              textShadow: `0 0 20px ${rewardColor}66`,
              animation: "number-pop .3s ease",
            }}>
              {reward >= 0 ? "+" : ""}{reward}
            </div>
            <Chip
              label={rewardLabel}
              color={rewardColor}
              bg={rewardColor + "18"}
            />
          </div>
        </PipeCard>

      </div>
    </div>
  );
}
