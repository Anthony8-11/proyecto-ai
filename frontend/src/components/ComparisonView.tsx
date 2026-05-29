import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ComparisonData } from "../hooks/useSimulation";

interface Props {
  comparison: ComparisonData | null;
  loading: boolean;
  error?: string | null;
  onCompare: () => void;
}

const AGENT_LABEL: Record<string, string> = {
  qlearning: "RL (Q-Learning)",
  dqn: "RL (DQN)",
};

const RL_COLOR = "#0ea5e9";
const ALGO_COLORS: Record<string, string> = {
  Greedy: "#f59e0b",
  "A*": "#a78bfa",
};

export function ComparisonView({ comparison, loading, error, onCompare }: Props) {
  const rlLabel = AGENT_LABEL[comparison?.rl.agent_type ?? "qlearning"] ?? "RL";

  const barData = comparison
    ? [
        { name: rlLabel, reward: comparison.rl.avg_reward_last10, isRL: true },
        { name: "Greedy", reward: comparison.greedy.avg_reward, isRL: false },
        { name: "A*", reward: comparison.astar.avg_reward, isRL: false },
      ]
    : [];

  const best = barData.length > 0 ? Math.max(...barData.map((d) => d.reward)) : null;

  function getColor(name: string): string {
    if (name === rlLabel) return RL_COLOR;
    return ALGO_COLORS[name] ?? "#64748b";
  }

  return (
    <div className="card">
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: "1rem",
        }}
      >
        <h3 style={{ margin: 0 }}>Comparación RL vs Búsqueda Clásica</h3>
        <button
          className="primary"
          style={{ padding: ".35rem .85rem", fontSize: ".8rem" }}
          onClick={onCompare}
          disabled={loading}
        >
          {loading ? "Calculando…" : "⟳ Comparar"}
        </button>
      </div>

      {error && !loading && (
        <div style={{
          background: "#2d050566", border: "1px solid #7f1d1d",
          borderRadius: 7, padding: ".6rem 1rem", marginBottom: ".75rem",
          fontSize: ".75rem", color: "#f87171",
        }}>
          Error: {error}
        </div>
      )}

      {!comparison && !loading && !error && (
        <p style={{ color: "#64748b", fontSize: ".85rem", textAlign: "center", padding: "2rem 0" }}>
          Inicia la simulación y haz clic en "Comparar" para ver el resultado.
        </p>
      )}

      {loading && (
        <p style={{ color: "#64748b", fontSize: ".85rem", textAlign: "center", padding: "2rem 0" }}>
          Ejecutando 10 episodios de Greedy y A*…
        </p>
      )}

      {comparison && !loading && (
        <>
          {/* Bar chart */}
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={barData} margin={{ top: 4, right: 8, left: -12, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis
                dataKey="name"
                tick={{ fill: "#94a3b8", fontSize: 11 }}
                axisLine={{ stroke: "#334155" }}
                tickLine={false}
              />
              <YAxis
                tick={{ fill: "#94a3b8", fontSize: 11 }}
                axisLine={{ stroke: "#334155" }}
                tickLine={false}
              />
              <Tooltip
                contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 6 }}
                labelStyle={{ color: "#e2e8f0" }}
                formatter={(v: number) => [`${v.toFixed(1)}`, "Reward promedio"]}
              />
              <ReferenceLine y={0} stroke="#475569" strokeDasharray="4 4" />
              <Bar dataKey="reward" radius={[4, 4, 0, 0]} maxBarSize={60}>
                {barData.map((entry) => (
                  <Cell key={entry.name} fill={getColor(entry.name)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          {/* Summary table */}
          <table
            style={{
              width: "100%",
              borderCollapse: "collapse",
              marginTop: "1rem",
              fontSize: ".82rem",
            }}
          >
            <thead>
              <tr style={{ borderBottom: "1px solid #1e293b" }}>
                <th style={{ textAlign: "left", padding: ".4rem .5rem", color: "#64748b" }}>Algoritmo</th>
                <th style={{ textAlign: "right", padding: ".4rem .5rem", color: "#64748b" }}>Reward promedio</th>
                <th style={{ textAlign: "right", padding: ".4rem .5rem", color: "#64748b" }}>vs RL</th>
              </tr>
            </thead>
            <tbody>
              {barData.map((row) => {
                const diff = row.reward - comparison.rl.avg_reward_last10;
                const isBest = row.reward === best;
                return (
                  <tr key={row.name} style={{ borderBottom: "1px solid #0f172a" }}>
                    <td style={{ padding: ".4rem .5rem", color: getColor(row.name) }}>
                      {isBest ? <span style={{ fontSize: ".52rem", fontWeight: 900, marginRight: 4, color: "inherit" }}>BEST</span> : null}{row.name}
                    </td>
                    <td style={{ textAlign: "right", padding: ".4rem .5rem", color: "#e2e8f0" }}>
                      {row.reward.toFixed(1)}
                    </td>
                    <td
                      style={{
                        textAlign: "right",
                        padding: ".4rem .5rem",
                        color: row.isRL ? "#64748b" : diff > 0 ? "#4ade80" : "#f87171",
                      }}
                    >
                      {row.isRL ? "—" : `${diff > 0 ? "+" : ""}${diff.toFixed(1)}`}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          <p style={{ fontSize: ".75rem", color: "#475569", marginTop: ".75rem" }}>
            {rlLabel}: últimos {Math.min(10, comparison.rl.total_episodes)} episodios · Greedy/A*: 10 episodios en entorno fresco
          </p>
        </>
      )}
    </div>
  );
}
