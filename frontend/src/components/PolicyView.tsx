import type { SimStep } from "../hooks/useSimulation";

const ACTION_LABELS = ["Redirigir flujo", "Activar generadores", "Monitorear", "Corte emergencia"];
const ACTION_TAGS   = ["RD", "GEN", "MON", "CRT"];
const ACTION_COLORS = ["#38bdf8", "#a78bfa", "#22c55e", "#fb923c"];

interface Props { step: SimStep | null; }

/** Blue → Yellow → Red heatmap */
function qColor(v: number, min: number, max: number): string {
  if (max === min) return "#1e293b";
  const t = (v - min) / (max - min);
  if (t < 0.5) {
    const u = t * 2;
    return `rgb(${Math.round(u*234)},${Math.round(u*179)},${Math.round(59+u*(8-59))})`;
  }
  const u = (t - 0.5) * 2;
  return `rgb(${Math.round(234+u*5)},${Math.round(179*(1-u))},8)`;
}

export function PolicyView({ step }: Props) {
  const qVals   = step?.q_values;
  const allowed = step?.allowed_actions ?? [];
  const type    = step?.agent_type ?? "qlearning";

  const min = qVals ? Math.min(...qVals) : 0;
  const max = qVals ? Math.max(...qVals) : 1;
  const best = qVals && allowed.length > 0
    ? allowed.reduce((b, a) => (qVals[a] > qVals[b] ? a : b), allowed[0])
    : null;

  return (
    <div className="card">
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: ".75rem" }}>
        <h3 style={{ margin: 0 }}>
          {type === "dqn" ? "Salida DQN — Logits" : "Tabla-Q — Estado actual"}
        </h3>
        <span style={{
          fontSize: ".58rem", padding: "1px 7px", borderRadius: 4,
          background: "#0b1a2e", border: "1px solid #1e4068",
          color: "#38bdf8", fontWeight: 700, textTransform: "uppercase", letterSpacing: ".06em",
        }}>
          {type === "dqn" ? "Red neuronal" : "Q-tabla 165×4"}
        </span>
      </div>

      {!qVals ? (
        <p style={{ fontSize: ".75rem", color: "#2a4060" }}>
          Sin datos — inicia la simulación.
        </p>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
          {ACTION_LABELS.map((lbl, a) => {
            const val       = qVals[a];
            const isAllowed = allowed.includes(a);
            const isBest    = a === best;
            const barColor  = isAllowed ? qColor(val, min, max) : "#162840";
            const barWidth  = isAllowed
              ? Math.max(8, ((val - min) / (max - min + 0.001)) * 100)
              : 0;

            return (
              <div key={a} style={{
                display: "flex", alignItems: "center", gap: 8,
                opacity: isAllowed ? 1 : 0.35,
                animation: "fade-up .25s ease",
              }}>
                {/* Action tag badge */}
                <span style={{
                  fontSize: ".52rem", fontWeight: 900, fontFamily: "monospace",
                  color: isAllowed ? ACTION_COLORS[a] : "#2a4060",
                  background: isAllowed ? ACTION_COLORS[a] + "18" : "transparent",
                  border: `1px solid ${isAllowed ? ACTION_COLORS[a] + "44" : "#162840"}`,
                  borderRadius: 3, padding: "1px 4px", minWidth: 26, textAlign: "center",
                }}>
                  {ACTION_TAGS[a]}
                </span>

                {/* Label */}
                <span style={{
                  fontSize: ".68rem", minWidth: 100,
                  color: isAllowed ? ACTION_COLORS[a] : "#2a4060",
                  fontWeight: isBest ? 800 : 500,
                }}>
                  {lbl}
                </span>

                {/* Bar track */}
                <div style={{
                  flex: 1, height: 22, borderRadius: 5,
                  background: "#0b1a2e",
                  border: isBest
                    ? "1.5px solid #38bdf8"
                    : `1px solid ${isAllowed ? "#1e4068" : "#162840"}`,
                  overflow: "hidden", position: "relative",
                }}>
                  {/* Fill */}
                  <div style={{
                    position: "absolute", left: 0, top: 0, bottom: 0,
                    width: `${barWidth}%`,
                    background: barColor,
                    borderRadius: 4,
                    transition: "width .4s cubic-bezier(.4,0,.2,1)",
                    opacity: isAllowed ? 0.85 : 0,
                  }}/>
                  {/* Value label */}
                  <span style={{
                    position: "absolute", left: 8, top: 0, bottom: 0,
                    display: "flex", alignItems: "center",
                    fontSize: ".65rem", fontWeight: 800,
                    color: isAllowed ? "#fff" : "#2a4060",
                    textShadow: isAllowed ? "0 1px 4px #0008" : "none",
                    zIndex: 1,
                  }}>
                    {val.toFixed(2)}
                  </span>
                </div>

                {/* Status indicator */}
                <span style={{
                  fontSize: ".6rem", fontWeight: 800, minWidth: 28, textAlign: "center",
                  color: isBest ? "#38bdf8" : isAllowed ? "transparent" : "#475569",
                  background: isBest ? "#38bdf822" : "transparent",
                  border: isBest ? "1px solid #38bdf844" : "1px solid transparent",
                  borderRadius: 3, padding: "1px 4px",
                  fontFamily: "monospace",
                }}>
                  {isBest ? "OPT" : isAllowed ? "" : "BLK"}
                </span>
              </div>
            );
          })}

          {/* Legend */}
          <div style={{
            marginTop: 4,
            display: "flex", gap: 14, paddingTop: 8,
            borderTop: "1px solid #162840",
          }}>
            <span style={{ fontSize: ".6rem", color: "#475569" }}>OPT — mejor acción permitida</span>
            <span style={{ fontSize: ".6rem", color: "#475569" }}>BLK — bloqueada por Prolog</span>
          </div>
        </div>
      )}
    </div>
  );
}
