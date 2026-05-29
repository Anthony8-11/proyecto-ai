import type { RuleInfo } from "../hooks/useSimulation";

const ACTION_LABELS = ["Redirigir flujo", "Activar generadores", "Monitorear", "Corte emergencia"];
const ACTION_TAGS   = ["RD", "GEN", "MON", "CRT"];
const ACTION_COLORS = ["#38bdf8", "#a78bfa", "#22c55e", "#fb923c"];

interface Props {
  rules: RuleInfo[] | undefined;
  logicEngine: string | undefined;
}

export function RulesPanel({ rules, logicEngine }: Props) {
  const activeCount = rules?.filter(r => r.active).length ?? 0;
  const grouped = [0, 1, 2, 3].map(action => ({
    action, label: ACTION_LABELS[action],
    tag: ACTION_TAGS[action], color: ACTION_COLORS[action],
    rules: rules?.filter(r => r.action === action) ?? [],
  }));

  return (
    <div className="card">
      {/* Header */}
      <div style={{
        display: "flex", alignItems: "center",
        justifyContent: "space-between", marginBottom: ".85rem",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: ".5rem" }}>
          <h3 style={{ margin: 0 }}>Motor Lógico — Reglas Prolog</h3>
        </div>
        <div style={{ display: "flex", gap: ".4rem", alignItems: "center" }}>
          {logicEngine === "prolog" ? (
            <span style={{
              fontSize: ".62rem", fontWeight: 700,
              background: "#052e16", color: "#4ade80",
              border: "1px solid #166534", borderRadius: 5, padding: "2px 8px",
            }}>
              ✓ SWI-Prolog
            </span>
          ) : logicEngine ? (
            <span style={{
              fontSize: ".62rem",
              background: "#1c1003", color: "#fbbf24",
              border: "1px solid #854d0e", borderRadius: 5, padding: "2px 8px",
            }}>
              ! Python fallback
            </span>
          ) : null}
          {rules && (
            <span style={{
              fontSize: ".62rem", fontWeight: 700, padding: "2px 8px",
              borderRadius: 5,
              background: activeCount > 0 ? "#2d050566" : "#052e1666",
              border: `1px solid ${activeCount > 0 ? "#7f1d1d" : "#166534"}`,
              color: activeCount > 0 ? "#f87171" : "#4ade80",
              animation: activeCount > 0 ? "blink-slow 2s infinite" : "none",
            }}>
              {activeCount > 0
                ? `${activeCount} activa${activeCount > 1 ? "s" : ""}`
                : "Sin restricciones"}
            </span>
          )}
        </div>
      </div>

      {!rules ? (
        <p style={{ fontSize: ".75rem", color: "#2a4060" }}>
          Inicia la simulación para ver las reglas en tiempo real.
        </p>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: ".75rem 1.25rem" }}>
          {grouped.map(({ action, label, tag, color, rules: gr }) => (
            <div key={action}>
              {/* Action header */}
              <div style={{
                display: "flex", alignItems: "center", gap: 6,
                marginBottom: 5,
              }}>
                <span style={{
                  fontSize: ".5rem", fontWeight: 900, fontFamily: "monospace",
                  color, background: color + "18",
                  border: `1px solid ${color}44`,
                  borderRadius: 3, padding: "1px 5px",
                }}>
                  {tag}
                </span>
                <span style={{
                  fontSize: ".6rem", fontWeight: 800, color,
                  textTransform: "uppercase", letterSpacing: ".07em",
                }}>
                  {label}
                </span>
              </div>

              {/* Rule chips */}
              <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                {gr.map(rule => (
                  <div key={rule.id} style={{
                    display: "flex", alignItems: "center", gap: 6,
                    background: rule.active ? "#2d050566" : "#0b1a2e",
                    border: `1px solid ${rule.active ? "#7f1d1d" : "#162840"}`,
                    borderRadius: 5, padding: "3px 7px",
                    transition: "background .25s, border-color .25s",
                    animation: rule.active ? "blink-slow 2s infinite" : "none",
                  }}>
                    {/* Status icon */}
                    <span style={{
                      fontSize: ".65rem", fontWeight: 900,
                      color: rule.active ? "#ef4444" : "#2a4060",
                      minWidth: 12,
                    }}>
                      {rule.active ? "✗" : "✓"}
                    </span>

                    {/* Rule ID */}
                    <span style={{
                      fontSize: ".58rem", fontWeight: 800, minWidth: 22,
                      color: rule.active ? "#f87171" : "#2a4060",
                      fontFamily: "monospace",
                    }}>
                      {rule.id}
                    </span>

                    {/* Description */}
                    <span style={{
                      fontSize: ".6rem",
                      color: rule.active ? "#fca5a5" : "#334155",
                    }}>
                      {rule.desc}
                    </span>

                    {/* Active glow dot */}
                    {rule.active && (
                      <div style={{
                        marginLeft: "auto", width: 5, height: 5, borderRadius: "50%",
                        background: "#ef4444", flexShrink: 0,
                        animation: "blink-fast .6s infinite",
                      }}/>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
