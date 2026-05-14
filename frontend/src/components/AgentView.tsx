import type { SimStep } from "../hooks/useSimulation";

const CONDITIONS = ["normal", "alert", "critical"] as const;
const CONDITION_LABELS = ["Normal", "Alerta", "Crítico"];
const ACTION_LABELS = ["Mover", "Asignar recursos", "Esperar", "Reaccionar"];

interface Props {
  step: SimStep | null;
  connected: boolean;
}

export function AgentView({ step, connected }: Props) {
  const condition = step ? CONDITIONS[step.state.env_condition] : "normal";

  return (
    <div className="card">
      <h3>Estado del agente</h3>

      <div style={{ marginBottom: ".75rem" }}>
        <span
          className={`badge-condition ${condition}`}
          style={{ marginRight: ".5rem" }}
        >
          {step ? CONDITION_LABELS[step.state.env_condition] : "—"}
        </span>
        <span style={{ fontSize: ".75rem", color: connected ? "#4ade80" : "#f87171" }}>
          {connected ? "● conectado" : "● desconectado"}
        </span>
      </div>

      <div className="stat-grid">
        <div className="stat">
          <div className="label">Posición</div>
          <div className="value">{step?.state.position ?? "—"}</div>
        </div>
        <div className="stat">
          <div className="label">Recursos</div>
          <div className="value">{step?.state.resources ?? "—"}</div>
        </div>
        <div className="stat">
          <div className="label">Episodio</div>
          <div className="value">{step?.episode ?? 1}</div>
        </div>
        <div className="stat">
          <div className="label">Reward acumulado</div>
          <div className="value" style={{ color: (step?.total_reward ?? 0) >= 0 ? "#4ade80" : "#f87171" }}>
            {step?.total_reward ?? 0}
          </div>
        </div>
      </div>

      <div style={{ marginTop: "1rem" }}>
        <div style={{ fontSize: ".75rem", color: "#64748b", marginBottom: ".3rem" }}>
          Última acción: <strong style={{ color: "#e2e8f0" }}>
            {step != null ? ACTION_LABELS[step.action] : "—"}
          </strong>
        </div>

        {/* Logic engine: allowed actions */}
        {step?.allowed_actions && (
          <div style={{ fontSize: ".72rem", color: "#64748b", marginBottom: ".4rem" }}>
            Acciones permitidas:{" "}
            {ACTION_LABELS.map((lbl, i) => (
              <span
                key={i}
                style={{
                  marginRight: ".25rem",
                  color: step.allowed_actions.includes(i) ? "#4ade80" : "#ef4444",
                  fontWeight: 600,
                }}
              >
                {lbl.slice(0, 3)}
              </span>
            ))}
          </div>
        )}

        {/* ML model prediction */}
        {step?.ml_probs && (
          <div style={{ fontSize: ".72rem", color: "#64748b", marginBottom: ".4rem" }}>
            ML predicción:{" "}
            <span style={{ color: "#fbbf24" }}>normal {((step.ml_probs[0] ?? 0) * 100).toFixed(0)}%</span>
            {" · "}
            <span style={{ color: "#fb923c" }}>alerta {((step.ml_probs[1] ?? 0) * 100).toFixed(0)}%</span>
            {" · "}
            <span style={{ color: "#f87171" }}>crítico {((step.ml_probs[2] ?? 0) * 100).toFixed(0)}%</span>
            {!step.ml_trained && (
              <span style={{ color: "#475569", marginLeft: ".3rem" }}>(sin entrenar)</span>
            )}
          </div>
        )}

        <div style={{ fontSize: ".75rem", color: "#64748b", marginBottom: ".5rem" }}>
          ε (exploración):{" "}
          <strong style={{ color: "#a78bfa" }}>{step?.epsilon ?? 1}</strong>
        </div>
        <div className="epsilon-bar">
          <div
            className="epsilon-fill"
            style={{ width: `${(step?.epsilon ?? 1) * 100}%` }}
          />
        </div>
      </div>

      {/* Agent grid */}
      <div style={{ marginTop: "1rem", display: "flex", gap: "6px" }}>
        {Array.from({ length: 5 }, (_, i) => (
          <div
            key={i}
            style={{
              flex: 1,
              height: 36,
              borderRadius: 6,
              background: step?.state.position === i ? "#0ea5e9" : "#0f172a",
              border: "1px solid #334155",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: ".7rem",
              color: step?.state.position === i ? "#fff" : "#475569",
              transition: "background .2s",
            }}
          >
            {step?.state.position === i ? "🤖" : i}
          </div>
        ))}
      </div>
    </div>
  );
}
