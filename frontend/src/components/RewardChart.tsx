import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SimStep } from "../hooks/useSimulation";

interface Props {
  history: SimStep[];
}

export function RewardChart({ history }: Props) {
  const data = history.map((s, i) => ({
    step: i,
    reward: s.reward,
    total: s.total_reward,
  }));

  return (
    <div className="card" style={{ flex: 1 }}>
      <h3>Recompensas por paso</h3>
      {data.length === 0 ? (
        <div style={{ color: "#475569", fontSize: ".85rem", marginTop: ".5rem" }}>
          Inicia la simulación para ver datos.
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="step" tick={{ fontSize: 10, fill: "#64748b" }} />
            <YAxis tick={{ fontSize: 10, fill: "#64748b" }} />
            <Tooltip
              contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 6 }}
              labelStyle={{ color: "#94a3b8", fontSize: 11 }}
            />
            <ReferenceLine y={0} stroke="#475569" strokeDasharray="4 4" />
            <Line
              type="monotone"
              dataKey="reward"
              stroke="#38bdf8"
              dot={false}
              strokeWidth={1.5}
              name="Reward paso"
            />
            <Line
              type="monotone"
              dataKey="total"
              stroke="#4ade80"
              dot={false}
              strokeWidth={1.5}
              name="Total episodio"
            />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
