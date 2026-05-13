import type { SimStep } from "../hooks/useSimulation";

const ACTION_LABELS = ["Mover", "Asignar", "Esperar", "Reaccionar"];
const CONDITION_EMOJI = ["🟢", "🟡", "🔴"];

interface Props {
  history: SimStep[];
}

export function DecisionLog({ history }: Props) {
  const recent = [...history].reverse().slice(0, 40);

  return (
    <div className="card">
      <h3>Historial de decisiones</h3>
      {recent.length === 0 ? (
        <div style={{ color: "#475569", fontSize: ".85rem" }}>Sin datos aún.</div>
      ) : (
        <ul className="log-list">
          {recent.map((s, i) => (
            <li key={i}>
              {CONDITION_EMOJI[s.state.env_condition]}{" "}
              <strong>{ACTION_LABELS[s.action]}</strong>
              {"  "}pos={s.state.position} res={s.state.resources}{"  "}
              <span className={s.reward >= 0 ? "pos-reward" : "neg-reward"}>
                {s.reward >= 0 ? "+" : ""}{s.reward}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
