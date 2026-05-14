import { AgentView } from "./components/AgentView";
import { ComparisonView } from "./components/ComparisonView";
import { DecisionLog } from "./components/DecisionLog";
import { RewardChart } from "./components/RewardChart";
import { useSimulation } from "./hooks/useSimulation";
import type { AgentType } from "./hooks/useSimulation";

export default function App() {
  const {
    connected, running, step, history, stats, comparison, loadingComparison,
    agentType, start, pause, reset, fetchStats, fetchComparison, switchAgent,
  } = useSimulation();

  return (
    <div className="app">
      <header>
        <h1>Sistema RL — Optimización Dinámica de Decisiones</h1>
        <span className="badge">Proyecto IA</span>
      </header>

      <main>
        {/* Sidebar */}
        <aside className="sidebar">
          <AgentView step={step} connected={connected} />

          {/* Agent selector */}
          <div className="card">
            <h3>Agente</h3>
            <div className="btn-row" style={{ marginBottom: ".75rem" }}>
              {(["qlearning", "dqn"] as AgentType[]).map((t) => (
                <button
                  key={t}
                  className={agentType === t ? "primary" : "secondary"}
                  style={{ flex: 1 }}
                  onClick={() => switchAgent(t)}
                  disabled={running}
                >
                  {t === "qlearning" ? "Q-Learning" : "DQN"}
                </button>
              ))}
            </div>
            <p style={{ fontSize: ".75rem", color: "#64748b", margin: 0 }}>
              {agentType === "qlearning"
                ? "Q-tabla tabular · espacio de estados discreto"
                : "Red neuronal · MLP 3→64→64→4 · replay buffer · target net"}
            </p>

            {/* DQN-specific telemetry */}
            {agentType === "dqn" && step?.dqn_loss !== undefined && (
              <div
                className="stat-grid"
                style={{ marginTop: ".75rem", gridTemplateColumns: "1fr 1fr" }}
              >
                <div className="stat">
                  <div className="label">Loss (MSE)</div>
                  <div className="value" style={{ fontSize: ".9rem", color: "#f59e0b" }}>
                    {step.dqn_loss.toFixed(4)}
                  </div>
                </div>
                <div className="stat">
                  <div className="label">Replay buffer</div>
                  <div className="value" style={{ fontSize: ".9rem", color: "#a78bfa" }}>
                    {step.dqn_buffer ?? 0}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Simulation controls */}
          <div className="card">
            <h3>Control</h3>
            <div className="btn-row">
              <button className="primary" onClick={start} disabled={running || !connected}>
                ▶ Iniciar
              </button>
              <button className="secondary" onClick={pause} disabled={!running}>
                ⏸ Pausar
              </button>
              <button className="danger" onClick={reset}>
                ↺ Reset
              </button>
            </div>
          </div>

          {/* Episode stats */}
          {stats && (
            <div className="card">
              <h3>Estadísticas</h3>
              <div className="stat-grid">
                <div className="stat">
                  <div className="label">Pasos totales</div>
                  <div className="value">{stats.steps}</div>
                </div>
                <div className="stat">
                  <div className="label">Episodios</div>
                  <div className="value">{stats.episodes}</div>
                </div>
                <div className="stat">
                  <div className="label">Media últimos 10</div>
                  <div className="value" style={{ fontSize: "1rem" }}>
                    {stats.mean_reward_last10}
                  </div>
                </div>
                <div className="stat">
                  <div className="label">Rec. promedio</div>
                  <div className="value" style={{ fontSize: "1rem" }}>
                    {stats.avg_resources ?? "—"}
                  </div>
                </div>
              </div>
              <button
                className="secondary"
                style={{ marginTop: ".75rem", width: "100%" }}
                onClick={fetchStats}
              >
                Actualizar stats
              </button>
            </div>
          )}
        </aside>

        {/* Main content */}
        <section className="content">
          <RewardChart history={history} />
          <ComparisonView
            comparison={comparison}
            loading={loadingComparison}
            onCompare={fetchComparison}
          />
          <DecisionLog history={history} />
        </section>
      </main>
    </div>
  );
}
