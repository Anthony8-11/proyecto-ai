import { AgentView } from "./components/AgentView";
import { ComparisonView } from "./components/ComparisonView";
import { DecisionLog } from "./components/DecisionLog";
import { RewardChart } from "./components/RewardChart";
import { useSimulation } from "./hooks/useSimulation";

export default function App() {
  const {
    connected, running, step, history, stats, comparison, loadingComparison,
    start, pause, reset, fetchStats, fetchComparison,
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
