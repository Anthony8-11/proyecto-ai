import { AgentView } from "./components/AgentView";
import { ComparisonView } from "./components/ComparisonView";
import { DecisionAnatomy } from "./components/DecisionAnatomy";
import { DecisionLog } from "./components/DecisionLog";
import { EpisodeChart } from "./components/EpisodeChart";
import { EnvironmentMap } from "./components/EnvironmentMap";
import { LeftSidebar } from "./components/LeftSidebar";
import { PolicyView } from "./components/PolicyView";
import { RewardChart } from "./components/RewardChart";
import { RightPanel } from "./components/RightPanel";
import { RulesPanel } from "./components/RulesPanel";
import { StartupModal } from "./components/StartupModal";
import { useSimulation } from "./hooks/useSimulation";
import type { EpisodeEntry, SimStats } from "./hooks/useSimulation";

export default function App() {
  const {
    connected, running, step, history, stats, comparison, loadingComparison, comparisonError,
    agentType, activeView, setActiveView, historicalData, saveStatus,
    showStartModal, startDataChecked,
    start, pause, reset, fetchStats, fetchComparison, switchAgent, saveState,
    resetFull, dismissStartModal, setAutoSaveInterval,
  } = useSimulation();

  const pauseAndSave = () => { pause(); saveState(); };

  const logicEngine = step?.logic_engine ?? null;
  const episodeLabel = step?.episode ?? 1;

  // Merge historical CSV data with current session episodes for the chart.
  // Each entry carries reward + agent_type so the chart can color per agent.
  // Capped at 200 historical + 50 session = 250 max data points.
  const mergedEpisodes: EpisodeEntry[] = stats
    ? [
        ...historicalData.slice(-200).map(h => ({
          reward:    h.total_reward,
          agentType: h.agent_type,
        })),
        ...stats.episode_rewards.map((r, i) => ({
          reward:    r,
          agentType: stats.episode_agents?.[i] ?? agentType,
        })),
      ]
    : [];

  // Keep mergedStats for components that still use SimStats (stats cards, etc.)
  const mergedStats: SimStats | null = stats ? {
    ...stats,
    episode_rewards: mergedEpisodes.map(e => e.reward),
  } : null;

  return (
    <div className="app-shell">

      {/* ── Top header ──────────────────────────────────────────── */}
      <header className="app-header">
        <div className="header-left" style={{ gap: ".75rem" }}>
          <span style={{ fontSize: ".72rem", fontWeight: 900, color: "var(--blue)", letterSpacing: ".08em" }}>
            SMART·GRID
          </span>
          <span className="badge" style={{ fontSize: ".52rem" }}>Proyecto IA</span>

          <span className={`conn-badge ${connected ? "ok" : "bad"}`} style={{ fontSize: ".55rem" }}>
            <span className="conn-dot"/>
            {connected ? "WS conectado" : "Sin conexión"}
          </span>
          {logicEngine === "prolog" && (
            <span style={{
              fontSize: ".55rem", padding: "2px 7px", borderRadius: 4,
              background: "#052e16", color: "#4ade80", border: "1px solid #166534", fontWeight: 700,
            }}>PROLOG OK</span>
          )}
          {logicEngine && logicEngine !== "prolog" && (
            <span style={{
              fontSize: ".55rem", padding: "2px 7px", borderRadius: 4,
              background: "#1c1003", color: "#fbbf24", border: "1px solid #854d0e", fontWeight: 600,
            }}>ML FALLBACK</span>
          )}
          {step?.ml_trained && (
            <span style={{
              fontSize: ".55rem", padding: "2px 7px", borderRadius: 4,
              background: "#052e16", color: "#4ade80", border: "1px solid #166534", fontWeight: 700,
            }}>ML OK</span>
          )}
        </div>

        <div className="header-right" style={{ gap: ".9rem" }}>
          <span style={{ fontSize: ".58rem", color: "var(--text-4)" }}>
            EP <span style={{ color: "var(--blue)", fontWeight: 800 }}>{episodeLabel}</span>
          </span>
          <span style={{ fontSize: ".58rem", color: "var(--text-4)" }}>
            SYS <span style={{ color: "var(--text-2)", fontWeight: 700 }}>
              {agentType === "dqn" ? "NEURO_DQN" : "Q_TABLE"}
            </span>
          </span>
          <span style={{ fontSize: ".58rem", color: "var(--text-4)" }}>
            ε <span style={{ color: "#a78bfa", fontWeight: 800 }}>{step?.epsilon ?? "1.0000"}</span>
          </span>
          <span style={{ fontSize: ".58rem", color: "var(--text-4)" }}>
            PASOS <span style={{ color: "var(--text-2)", fontWeight: 700 }}>{history.length}</span>
          </span>
        </div>
      </header>

      {/* ── Left sidebar ────────────────────────────────────────── */}
      <LeftSidebar
        step={step}
        stats={stats}
        activeView={activeView}
        onNavigate={setActiveView}
        connected={connected}
      />

      {/* ── Main content (switches by activeView) ───────────────── */}
      <main className="main-content">

        {/* OPERACIONES */}
        {activeView === "operaciones" && (
          <>
            <EnvironmentMap step={step}/>
            <DecisionAnatomy step={step}/>
            <DecisionLog history={history}/>
          </>
        )}

        {/* ENTRENAMIENTO */}
        {activeView === "entrenamiento" && (
          <>
            <AgentView step={step} connected={connected}/>
            <PolicyView step={step}/>
            <div className="card">
              <h3>Tipo de Agente RL</h3>
              <div className="btn-row" style={{ marginBottom: ".7rem" }}>
                {(["qlearning", "dqn"] as const).map(t => (
                  <button
                    key={t}
                    className={agentType === t ? "primary" : "secondary"}
                    style={{ flex: 1, fontSize: ".72rem" }}
                    onClick={() => switchAgent(t)}
                    disabled={running}
                    type="button"
                  >
                    {t === "qlearning" ? "Q-Learning" : "DQN"}
                  </button>
                ))}
              </div>
              <p style={{ fontSize: ".68rem", color: "var(--text-3)", margin: 0 }}>
                {agentType === "qlearning"
                  ? "Q-tabla 165×4 · ε-greedy · Bellman update"
                  : "MLP 3→64→64→4 · replay buffer · target net"}
              </p>
              {agentType === "dqn" && step?.dqn_loss !== undefined && (
                <div className="stat-grid" style={{ marginTop: ".65rem" }}>
                  <div className="stat">
                    <div className="label">Loss (MSE)</div>
                    <div className="value" style={{ fontSize: ".9rem", color: "#f59e0b" }}>
                      {step.dqn_loss.toFixed(4)}
                    </div>
                  </div>
                  <div className="stat">
                    <div className="label">Buffer</div>
                    <div className="value" style={{ fontSize: ".9rem", color: "#a78bfa" }}>
                      {step.dqn_buffer ?? 0}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </>
        )}

        {/* REGLAS PROLOG */}
        {activeView === "reglas" && (
          <RulesPanel rules={step?.rules_info} logicEngine={step?.logic_engine}/>
        )}

        {/* COMPARACIÓN */}
        {activeView === "comparacion" && (
          <ComparisonView
            comparison={comparison}
            loading={loadingComparison}
            error={comparisonError}
            onCompare={fetchComparison}
          />
        )}

        {/* HISTÓRICO */}
        {activeView === "historico" && (
          <>
            <RewardChart history={history}/>
            <EpisodeChart stats={mergedStats} episodes={mergedEpisodes}/>
            {stats && (
              <div className="card">
                <h3>Estadísticas de sesión</h3>
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
                    <div className="label">Media últ. 10</div>
                    <div className="value" style={{ fontSize: ".95rem" }}>
                      {stats.mean_reward_last10 ?? 0}
                    </div>
                  </div>
                  <div className="stat">
                    <div className="label">Cap. promedio</div>
                    <div className="value" style={{ fontSize: ".95rem" }}>
                      {stats.avg_resources ?? "—"}
                    </div>
                  </div>
                </div>
                <div style={{ display: "flex", gap: ".5rem", marginTop: ".6rem" }}>
                  <button
                    className="secondary"
                    style={{ flex: 1, fontSize: ".72rem" }}
                    onClick={fetchStats}
                    type="button"
                  >
                    Actualizar
                  </button>
                  <button
                    className="secondary"
                    style={{
                      flex: 1, fontSize: ".72rem",
                      color: saveStatus === "ok" ? "#4ade80" : saveStatus === "error" ? "#f87171" : undefined,
                    }}
                    onClick={saveState}
                    disabled={saveStatus === "saving"}
                    type="button"
                  >
                    {saveStatus === "saving" ? "Guardando…" : saveStatus === "ok" ? "Guardado ✓" : "Guardar"}
                  </button>
                </div>
              </div>
            )}
          </>
        )}

      </main>

      {/* ── Right panel ─────────────────────────────────────────── */}
      <RightPanel
        step={step}
        history={history}
        stats={stats}
        running={running}
        connected={connected}
        agentType={agentType}
        onStart={start}
        onPause={pause}
        onReset={reset}
        onSwitchAgent={switchAgent}
        onPauseAndSave={pauseAndSave}
        onSetAutoSaveInterval={setAutoSaveInterval}
        saveStatus={saveStatus}
      />

      {/* ── Startup modal (choose saved vs fresh) ───────────────── */}
      {showStartModal && (
        <StartupModal
          checked={startDataChecked}
          historicalData={historicalData}
          stats={stats}
          onContinue={dismissStartModal}
          onReset={async () => {
            await resetFull();
            dismissStartModal();
          }}
        />
      )}

    </div>
  );
}
