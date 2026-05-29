import { useCallback, useEffect, useRef, useState } from "react";

export interface RuleInfo {
  id: string;
  action: number;
  desc: string;
  active: boolean;
}

export interface SimStep {
  state: { position: number; resources: number; env_condition: number; index: number };
  action: number;
  allowed_actions: number[];
  reward: number;
  done: boolean;
  episode: number;
  total_reward: number;
  epsilon: number;
  ml_probs: Record<number, number>;
  ml_trained: boolean;
  agent_type: "qlearning" | "dqn";
  logic_engine?: string;
  dqn_loss?: number;
  dqn_buffer?: number;
  q_values?: number[];
  rules_info?: RuleInfo[];
  decision_mode?: string;
}

export type AgentType = "qlearning" | "dqn";

/** One episode's data for the learning curve chart. */
export interface EpisodeEntry {
  reward: number;
  agentType: string;
}
export type ActiveView = "operaciones" | "entrenamiento" | "reglas" | "comparacion" | "historico";

export interface HistoricalEpisode {
  episode_num: number;
  total_reward: number;
  steps: number;
  epsilon: number;
  agent_type: string;
}

export interface ComparisonData {
  rl: {
    avg_reward_last10: number;
    total_episodes: number;
    episode_rewards: number[];
    agent_type: string;
  };
  greedy: { avg_reward: number };
  astar: { avg_reward: number };
}

export interface SimStats {
  steps: number;
  episodes: number;
  episode_rewards: number[];
  episode_agents?: string[];   // parallel array: agent type per episode
  mean_reward_last10: number;
  action_distribution: Record<number, number>;
  avg_resources?: number;
  condition_distribution?: Record<number, number>;
  last_saved_episode?: number;
  auto_save_every?: number;
}

const ACTION_LABELS = ["Mover", "Asignar recursos", "Esperar", "Reaccionar"];
// Use Vite proxy — avoids CORS entirely. Relative /api goes through Vite → backend.
const WS_URL = `ws://${window.location.host}/api/ws`;
const API_BASE = "/api";

/** Exponential-backoff reconnect: 1s → 2s → 4s → … → 30s cap. */
const backoffDelay = (attempt: number) =>
  Math.min(1000 * 2 ** attempt, 30_000);

export function useSimulation() {
  const ws = useRef<WebSocket | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const historyFlushRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectAttemptRef = useRef(0);
  const mountedRef = useRef(true);
  // Accumulates incoming steps between 500ms history flushes
  const historyBatchRef = useRef<SimStep[]>([]);
  // Holds last full step so optional fields (q_values, rules_info) are not lost
  const lastFullStepRef = useRef<SimStep | null>(null);

  const [connected, setConnected] = useState(false);
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState<SimStep | null>(null);
  const [history, setHistory] = useState<SimStep[]>([]);
  const [stats, setStats] = useState<SimStats | null>(null);
  const [comparison, setComparison] = useState<ComparisonData | null>(null);
  const [loadingComparison, setLoadingComparison] = useState(false);
  const [agentType, setAgentTypeState] = useState<AgentType>("qlearning");
  const [activeView, setActiveView] = useState<ActiveView>("operaciones");
  const [historicalData, setHistoricalData] = useState<HistoricalEpisode[]>([]);
  const [saveStatus, setSaveStatus] = useState<"idle" | "saving" | "ok" | "error">("idle");
  const [comparisonError, setComparisonError] = useState<string | null>(null);
  const [showStartModal, setShowStartModal] = useState(true);
  const [startDataChecked, setStartDataChecked] = useState(false);

  // ------------------------------------------------------------------
  // WebSocket connection with automatic exponential-backoff reconnect
  // ------------------------------------------------------------------
  const connect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }

    if (ws.current) {
      ws.current.onclose = null;
      ws.current.close();
    }

    const socket = new WebSocket(WS_URL);

    socket.onopen = () => {
      if (!mountedRef.current) return;
      setConnected(true);
      reconnectAttemptRef.current = 0;
    };

    socket.onclose = () => {
      if (!mountedRef.current) return;
      setConnected(false);
      setRunning(false);
      const delay = backoffDelay(reconnectAttemptRef.current);
      reconnectAttemptRef.current += 1;
      reconnectTimerRef.current = setTimeout(connect, delay);
    };

    socket.onmessage = (e) => {
      if (!mountedRef.current) return;
      const incoming: SimStep = JSON.parse(e.data);

      // Merge with the last full step so optional fields (q_values, rules_info)
      // retain their last known value when the backend skips them to save bandwidth.
      const prev = lastFullStepRef.current;
      const data: SimStep = prev
        ? {
            ...incoming,
            q_values:   incoming.q_values   ?? prev.q_values,
            rules_info: incoming.rules_info ?? prev.rules_info,
          }
        : incoming;
      lastFullStepRef.current = data;

      // step updates immediately (current-state display)
      setStep(data);

      // history updates are batched and flushed every 500ms to reduce
      // the number of React re-renders on chart/sparkline components.
      historyBatchRef.current.push(data);
    };

    ws.current = socket;
  }, []); // stable ref — uses only refs internally

  const sendStep = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ action: "step" }));
    }
  }, []);

  // ------------------------------------------------------------------
  // Simulation controls
  // ------------------------------------------------------------------
  const start = useCallback(() => {
    if (!connected) connect();
    setRunning(true);
  }, [connected, connect]);

  const pause = useCallback(() => setRunning(false), []);

  const reset = useCallback(async () => {
    setRunning(false);
    setHistory([]);
    setStep(null);
    await fetch(`${API_BASE}/simulation/reset`, { method: "POST" });
    const res = await fetch(`${API_BASE}/simulation/stats`);
    if (res.ok) setStats(await res.json());
  }, []);

  const fetchStats = useCallback(async () => {
    const res = await fetch(`${API_BASE}/simulation/stats`);
    if (res.ok) setStats(await res.json());
  }, []);

  const fetchComparison = useCallback(async () => {
    setLoadingComparison(true);
    setComparisonError(null);
    try {
      const res = await fetch(`${API_BASE}/simulation/comparison`);
      if (res.ok) {
        setComparison(await res.json());
      } else {
        setComparisonError(`Error ${res.status}: ${res.statusText}`);
      }
    } catch (e) {
      setComparisonError("No se pudo conectar con el backend.");
    } finally {
      setLoadingComparison(false);
    }
  }, []);

  const switchAgent = useCallback(async (type: AgentType) => {
    setRunning(false);
    setStep(null);          // clear current-step display only
    setAgentTypeState(type);
    await fetch(`${API_BASE}/simulation/agent-type`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type }),
    });
    // Refresh stats so the panel reflects the continuing session
    const res = await fetch(`${API_BASE}/simulation/stats`);
    if (res.ok) setStats(await res.json());
  }, []);

  const saveState = useCallback(async () => {
    setSaveStatus("saving");
    try {
      const res = await fetch(`${API_BASE}/simulation/save`, { method: "POST" });
      if (res.ok) {
        setSaveStatus("ok");
        setTimeout(() => setSaveStatus("idle"), 2500);
      } else {
        setSaveStatus("error");
        setTimeout(() => setSaveStatus("idle"), 3000);
      }
    } catch {
      setSaveStatus("error");
      setTimeout(() => setSaveStatus("idle"), 3000);
    }
  }, []);

  const fetchHistoricalData = useCallback(async () => {
    const res = await fetch(`${API_BASE}/training/history`);
    if (res.ok) {
      const data = await res.json();
      setHistoricalData(data.episodes ?? []);
    }
  }, []);

  /** Change the auto-save interval (1–20 episodes). */
  const setAutoSaveInterval = useCallback(async (every: number) => {
    await fetch(`${API_BASE}/simulation/autosave-interval`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ every }),
    });
    // Refresh stats so the panel reflects the new interval immediately
    const res = await fetch(`${API_BASE}/simulation/stats`);
    if (res.ok) setStats(await res.json());
  }, []);

  /** Wipe all persisted training data on the backend and reset local state. */
  const resetFull = useCallback(async () => {
    await fetch(`${API_BASE}/simulation/reset-full`, { method: "POST" });
    setHistory([]);
    setStep(null);
    setStats(null);
    setHistoricalData([]);
    setRunning(false);
    // Refresh stats so UI reflects zero state
    const res = await fetch(`${API_BASE}/simulation/stats`);
    if (res.ok) setStats(await res.json());
  }, []);

  const dismissStartModal = useCallback(() => setShowStartModal(false), []);

  // ------------------------------------------------------------------
  // Effects
  // ------------------------------------------------------------------

  // Auto-step loop
  useEffect(() => {
    if (running) {
      intervalRef.current = setInterval(sendStep, 150);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [running, sendStep]);

  // Poll stats every 3s while running
  useEffect(() => {
    if (!running) return;
    const id = setInterval(fetchStats, 3000);
    return () => clearInterval(id);
  }, [running, fetchStats]);

  // Connect on mount, initial fetches, cleanup on unmount
  useEffect(() => {
    mountedRef.current = true;
    connect();
    // Wait for both initial fetches to finish before unlocking the modal
    Promise.all([fetchStats(), fetchHistoricalData()]).then(() => {
      if (mountedRef.current) setStartDataChecked(true);
    });

    // History flush — drains historyBatchRef into React state every 500 ms.
    // Reduces re-renders of chart/sparkline components from ~6.7/s to ~2/s.
    historyFlushRef.current = setInterval(() => {
      if (!mountedRef.current) return;
      const batch = historyBatchRef.current;
      if (batch.length === 0) return;
      historyBatchRef.current = [];
      setHistory((h) => {
        const combined = [...h, ...batch];
        return combined.length > 200 ? combined.slice(-200) : combined;
      });
    }, 500);

    return () => {
      mountedRef.current = false;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (historyFlushRef.current) clearInterval(historyFlushRef.current);
      if (ws.current) {
        ws.current.onclose = null;
        ws.current.close();
      }
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps — intentionally run once

  return {
    connected,
    running,
    step,
    history,
    stats,
    comparison,
    loadingComparison,
    comparisonError,
    agentType,
    activeView,
    setActiveView,
    historicalData,
    saveStatus,
    showStartModal,
    startDataChecked,
    start,
    pause,
    reset,
    fetchStats,
    fetchComparison,
    switchAgent,
    saveState,
    fetchHistoricalData,
    resetFull,
    dismissStartModal,
    setAutoSaveInterval,
    ACTION_LABELS,
  };
}
