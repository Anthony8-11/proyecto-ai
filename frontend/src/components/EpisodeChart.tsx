import { memo } from "react";
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
import type { EpisodeEntry, SimStats } from "../hooks/useSimulation";

interface Props {
  stats: SimStats | null;
  episodes?: EpisodeEntry[];   // rich per-episode data (reward + agentType)
}

const COLOR_QL  = "#38bdf8";   // cyan   — Q-Learning
const COLOR_DQN = "#a78bfa";   // purple — DQN
const COLOR_AVG = "#f59e0b";   // amber  — moving average

const AGENT_LABEL: Record<string, string> = {
  qlearning: "Q-Learning",
  dqn:       "DQN",
};

/** N-point moving average; returns undefined for the first (window-1) entries. */
function movingAverage(values: number[], window: number): (number | undefined)[] {
  return values.map((_, i) => {
    if (i < window - 1) return undefined;
    const slice = values.slice(i - window + 1, i + 1);
    return Math.round(slice.reduce((a, b) => a + b, 0) / window);
  });
}

/** Detect episode indices where the agent type changes. */
function switchPoints(entries: EpisodeEntry[]): { episode: number; from: string; to: string }[] {
  const pts: { episode: number; from: string; to: string }[] = [];
  for (let i = 1; i < entries.length; i++) {
    if (entries[i].agentType !== entries[i - 1].agentType) {
      pts.push({ episode: i + 1, from: entries[i - 1].agentType, to: entries[i].agentType });
    }
  }
  return pts;
}

export const EpisodeChart = memo(function EpisodeChart({ stats, episodes = [] }: Props) {
  // Fall back to flat stats if rich episodes not provided
  const rewards = episodes.length > 0
    ? episodes.map(e => e.reward)
    : (stats?.episode_rewards ?? []);

  const avg5   = movingAverage(rewards, 5);
  const switches = switchPoints(episodes);

  const data = rewards.map((r, i) => {
    const agentType = episodes[i]?.agentType ?? "qlearning";
    return {
      episode:    i + 1,
      reward_ql:  agentType === "qlearning" ? Math.round(r) : null,
      reward_dqn: agentType === "dqn"       ? Math.round(r) : null,
      avg5:       avg5[i],
    };
  });

  const hasData = data.length > 0;
  const hasQL  = episodes.some(e => e.agentType === "qlearning");
  const hasDQN = episodes.some(e => e.agentType === "dqn");

  // Trend indicator
  let trend: "subiendo" | "estable" | "bajando" | null = null;
  if (rewards.length >= 6) {
    const half = Math.floor(rewards.length / 2);
    const first = rewards.slice(0, half).reduce((a, b) => a + b, 0) / half;
    const second = rewards.slice(half).reduce((a, b) => a + b, 0) / (rewards.length - half);
    const delta = second - first;
    trend = delta > 20 ? "subiendo" : delta < -20 ? "bajando" : "estable";
  }

  const trendColor = trend === "subiendo" ? "#4ade80" : trend === "bajando" ? "#f87171" : "#94a3b8";
  const trendIcon  = trend === "subiendo" ? "↑" : trend === "bajando" ? "↓" : "→";

  // Label for the switch ReferenceLine
  const switchLabel = (from: string, to: string) =>
    `${AGENT_LABEL[from] ?? from} → ${AGENT_LABEL[to] ?? to}`;

  return (
    <div className="card" style={{ flex: 1 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: ".5rem" }}>
        <h3 style={{ margin: 0 }}>Curva de Aprendizaje — Reward por Episodio</h3>
        {trend && (
          <span style={{ fontSize: ".78rem", color: trendColor, fontWeight: 700 }}>
            {trendIcon} {trend}
          </span>
        )}
      </div>

      {!hasData ? (
        <div style={{ color: "#475569", fontSize: ".85rem", marginTop: ".5rem" }}>
          Completa al menos un episodio para ver la curva de convergencia.
        </div>
      ) : (
        <>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis
                dataKey="episode"
                tick={{ fontSize: 10, fill: "#64748b" }}
                label={{ value: "Episodio", position: "insideBottomRight", offset: -4, fontSize: 10, fill: "#475569" }}
              />
              <YAxis tick={{ fontSize: 10, fill: "#64748b" }} />
              <Tooltip
                contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 6 }}
                labelStyle={{ color: "#94a3b8", fontSize: 11 }}
                labelFormatter={(v) => `Episodio ${v}`}
                formatter={(value, name) => {
                  const labels: Record<string, string> = {
                    reward_ql:  "Q-Learning",
                    reward_dqn: "DQN",
                    avg5:       "Media 5 ep.",
                  };
                  const v = value as number | null;
                  return [v != null ? v : "—", labels[name as string] ?? name] as [string | number, string];
                }}
              />
              <ReferenceLine y={0} stroke="#475569" strokeDasharray="4 4" />

              {/* Vertical markers at agent-switch episodes */}
              {switches.map((sw) => (
                <ReferenceLine
                  key={sw.episode}
                  x={sw.episode}
                  stroke="#475569"
                  strokeDasharray="3 3"
                  strokeWidth={1.5}
                  label={{
                    value:    switchLabel(sw.from, sw.to),
                    position: "insideTopRight",
                    fill:     "#64748b",
                    fontSize: 9,
                  }}
                />
              ))}

              {/* Q-Learning line */}
              {hasQL && (
                <Line
                  type="monotone"
                  dataKey="reward_ql"
                  stroke={COLOR_QL}
                  dot={data.length <= 30}
                  strokeWidth={1.5}
                  name="reward_ql"
                  connectNulls={false}
                />
              )}

              {/* DQN line */}
              {hasDQN && (
                <Line
                  type="monotone"
                  dataKey="reward_dqn"
                  stroke={COLOR_DQN}
                  dot={data.length <= 30}
                  strokeWidth={1.5}
                  name="reward_dqn"
                  connectNulls={false}
                />
              )}

              {/* Moving average across all episodes */}
              <Line
                type="monotone"
                dataKey="avg5"
                stroke={COLOR_AVG}
                dot={false}
                strokeWidth={2}
                strokeDasharray="5 3"
                name="avg5"
                connectNulls={false}
              />
            </LineChart>
          </ResponsiveContainer>

          {/* Legend */}
          <div style={{ display: "flex", gap: "1.25rem", marginTop: ".5rem", fontSize: ".72rem", color: "#64748b", flexWrap: "wrap" }}>
            {hasQL && (
              <span>
                <span style={{ color: COLOR_QL }}>—</span> Q-Learning
              </span>
            )}
            {hasDQN && (
              <span>
                <span style={{ color: COLOR_DQN }}>—</span> DQN
              </span>
            )}
            <span>
              <span style={{ color: COLOR_AVG }}>- -</span> Media móvil 5 ep.
            </span>
            {switches.length > 0 && (
              <span>
                <span style={{ color: "#475569" }}>|</span> Cambio de agente
              </span>
            )}
            <span style={{ marginLeft: "auto" }}>
              {rewards.length} ep. · mejor:{" "}
              <span style={{ color: "#4ade80" }}>{Math.max(...rewards).toFixed(0)}</span>
            </span>
          </div>
        </>
      )}
    </div>
  );
});
