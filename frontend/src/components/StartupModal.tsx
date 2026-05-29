import type { HistoricalEpisode, SimStats } from "../hooks/useSimulation";

interface Props {
  checked: boolean;           // true once initial data fetch is done
  historicalData: HistoricalEpisode[];
  stats: SimStats | null;
  onContinue: () => void;
  onReset: () => void;
}

export function StartupModal({ checked, historicalData, stats, onContinue, onReset }: Props) {
  const totalHistoricEpisodes = historicalData.length;
  const totalHistoricSteps = historicalData.reduce((s, e) => s + e.steps, 0);
  const hasSavedData = checked && (totalHistoricEpisodes > 0 || (stats?.steps ?? 0) > 0);
  const lastEpsilon = historicalData.length > 0
    ? historicalData[historicalData.length - 1].epsilon
    : null;
  const lastAgent = historicalData.length > 0
    ? historicalData[historicalData.length - 1].agent_type
    : null;

  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 9999,
      background: "rgba(2, 6, 15, 0.88)",
      display: "flex", alignItems: "center", justifyContent: "center",
      backdropFilter: "blur(6px)",
    }}>
      <div style={{
        background: "var(--surface-1, #0f172a)",
        border: "1px solid #1e293b",
        borderRadius: 12,
        padding: "2rem 2.25rem",
        width: "min(90vw, 440px)",
        boxShadow: "0 24px 64px rgba(0,0,0,.6)",
        display: "flex", flexDirection: "column", gap: "1.25rem",
      }}>

        {/* Header */}
        <div>
          <div style={{
            fontSize: ".65rem", fontWeight: 900, letterSpacing: ".15em",
            color: "var(--blue, #0ea5e9)", marginBottom: ".35rem",
          }}>
            SMART·GRID
          </div>
          <h2 style={{ margin: 0, fontSize: "1.15rem", fontWeight: 800, color: "#e2e8f0" }}>
            Iniciar sesión de entrenamiento
          </h2>
          <p style={{ margin: ".4rem 0 0", fontSize: ".78rem", color: "#64748b" }}>
            Elige cómo deseas comenzar esta sesión.
          </p>
        </div>

        {/* Saved data summary */}
        <div style={{
          background: "#0a1628",
          border: "1px solid #1e293b",
          borderRadius: 8,
          padding: "1rem 1.1rem",
        }}>
          <div style={{
            fontSize: ".6rem", fontWeight: 700, letterSpacing: ".1em",
            color: "#475569", textTransform: "uppercase", marginBottom: ".65rem",
          }}>
            Datos almacenados
          </div>

          {!checked ? (
            <div style={{ fontSize: ".78rem", color: "#475569" }}>
              Verificando...
            </div>
          ) : hasSavedData ? (
            <div style={{ display: "flex", flexDirection: "column", gap: ".45rem" }}>
              <Row label="Episodios guardados" value={String(totalHistoricEpisodes)} color="#22c55e" />
              <Row label="Pasos totales" value={String(totalHistoricSteps.toLocaleString())} color="#38bdf8" />
              {lastEpsilon !== null && (
                <Row label="Epsilon guardado" value={lastEpsilon.toFixed(4)} color="#a78bfa" />
              )}
              {lastAgent && (
                <Row
                  label="Agente"
                  value={lastAgent === "dqn" ? "DQN" : "Q-Learning"}
                  color="#f59e0b"
                />
              )}
            </div>
          ) : (
            <div style={{
              fontSize: ".78rem", color: "#475569",
              display: "flex", alignItems: "center", gap: ".5rem",
            }}>
              <span style={{
                display: "inline-block", width: 6, height: 6, borderRadius: "50%",
                background: "#475569", flexShrink: 0,
              }} />
              No se encontraron datos de sesiones anteriores
            </div>
          )}
        </div>

        {/* Action cards */}
        <div style={{ display: "flex", flexDirection: "column", gap: ".65rem" }}>

          {/* Continue card */}
          <ActionCard
            title="Continuar entrenamiento"
            description={
              hasSavedData
                ? `Retomar desde el episodio ${totalHistoricEpisodes} con el estado guardado`
                : "Iniciar con agente sin entrenar (no hay datos guardados)"
            }
            tag={hasSavedData ? "RECOMENDADO" : undefined}
            tagColor="#22c55e"
            disabled={!checked}
            onClick={onContinue}
            primary
          />

          {/* Reset card */}
          <ActionCard
            title="Iniciar desde cero"
            description="Eliminar todo el historial y entrenar el agente desde el estado inicial"
            disabled={!checked}
            onClick={onReset}
            primary={false}
            danger
          />
        </div>

        <p style={{
          margin: 0, fontSize: ".68rem", color: "#334155",
          borderTop: "1px solid #1e293b", paddingTop: ".9rem",
        }}>
          Esta eleccion solo afecta la sesion actual. Los datos se pueden guardar manualmente
          desde el panel de control.
        </p>

      </div>
    </div>
  );
}

/* ── Helper components ─────────────────────────────────────── */

function Row({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
      <span style={{ fontSize: ".73rem", color: "#64748b" }}>{label}</span>
      <span style={{ fontSize: ".78rem", fontWeight: 700, color }}>{value}</span>
    </div>
  );
}

interface ActionCardProps {
  title: string;
  description: string;
  tag?: string;
  tagColor?: string;
  disabled?: boolean;
  onClick: () => void;
  primary?: boolean;
  danger?: boolean;
}

function ActionCard({ title, description, tag, tagColor, disabled, onClick, primary, danger }: ActionCardProps) {
  const borderColor = danger ? "#2d1a1a" : primary ? "#0c2a3f" : "#1e293b";
  const hoverBorderColor = danger ? "#7f1d1d" : primary ? "#0ea5e9" : "#334155";

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      type="button"
      style={{
        background: danger ? "#1a0a0a" : primary ? "#071828" : "transparent",
        border: `1px solid ${borderColor}`,
        borderRadius: 8,
        padding: ".85rem 1rem",
        cursor: disabled ? "not-allowed" : "pointer",
        textAlign: "left",
        opacity: disabled ? 0.5 : 1,
        transition: "border-color .15s, background .15s",
        width: "100%",
      }}
      onMouseEnter={e => {
        if (!disabled) (e.currentTarget as HTMLButtonElement).style.borderColor = hoverBorderColor;
      }}
      onMouseLeave={e => {
        (e.currentTarget as HTMLButtonElement).style.borderColor = borderColor;
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: ".5rem", marginBottom: ".3rem" }}>
        <span style={{
          fontSize: ".82rem", fontWeight: 700,
          color: danger ? "#f87171" : primary ? "#e2e8f0" : "#94a3b8",
        }}>
          {title}
        </span>
        {tag && (
          <span style={{
            fontSize: ".52rem", fontWeight: 900, letterSpacing: ".08em",
            color: tagColor, border: `1px solid ${tagColor}`,
            borderRadius: 4, padding: "1px 5px",
          }}>
            {tag}
          </span>
        )}
      </div>
      <p style={{ margin: 0, fontSize: ".7rem", color: "#475569", lineHeight: 1.4 }}>
        {description}
      </p>
    </button>
  );
}
