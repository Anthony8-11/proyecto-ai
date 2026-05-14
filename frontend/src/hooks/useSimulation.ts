import { useCallback, useEffect, useRef, useState } from "react";

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
}

export interface ComparisonData {
  rl: { avg_reward_last10: number; total_episodes: number; episode_rewards: number[] };
  greedy: { avg_reward: number };
  astar: { avg_reward: number };
}

export interface SimStats {
  steps: number;
  episodes: number;
  episode_rewards: number[];
  mean_reward_last10: number;
  action_distribution: Record<number, number>;
  avg_resources?: number;
  condition_distribution?: Record<number, number>;
}

const ACTION_LABELS = ["Mover", "Asignar recursos", "Esperar", "Reaccionar"];
const WS_URL = "ws://localhost:8000/api/ws";

export function useSimulation() {
  const ws = useRef<WebSocket | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [connected, setConnected] = useState(false);
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState<SimStep | null>(null);
  const [history, setHistory] = useState<SimStep[]>([]);
  const [stats, setStats] = useState<SimStats | null>(null);
  const [comparison, setComparison] = useState<ComparisonData | null>(null);
  const [loadingComparison, setLoadingComparison] = useState(false);

  const connect = useCallback(() => {
    if (ws.current) ws.current.close();
    const socket = new WebSocket(WS_URL);
    socket.onopen = () => setConnected(true);
    socket.onclose = () => { setConnected(false); setRunning(false); };
    socket.onmessage = (e) => {
      const data: SimStep = JSON.parse(e.data);
      setStep(data);
      setHistory((h) => [...h.slice(-199), data]);
    };
    ws.current = socket;
  }, []);

  const sendStep = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ action: "step" }));
    }
  }, []);

  const start = useCallback(() => {
    if (!connected) connect();
    setRunning(true);
  }, [connected, connect]);

  const pause = useCallback(() => setRunning(false), []);

  const reset = useCallback(async () => {
    setRunning(false);
    setHistory([]);
    setStep(null);
    setStats(null);
    await fetch("/api/simulation/reset", { method: "POST" });
  }, []);

  const fetchStats = useCallback(async () => {
    const res = await fetch("/api/simulation/stats");
    if (res.ok) setStats(await res.json());
  }, []);

  const fetchComparison = useCallback(async () => {
    setLoadingComparison(true);
    try {
      const res = await fetch("/api/simulation/comparison");
      if (res.ok) setComparison(await res.json());
    } finally {
      setLoadingComparison(false);
    }
  }, []);

  // Auto-step loop
  useEffect(() => {
    if (running) {
      intervalRef.current = setInterval(sendStep, 150);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [running, sendStep]);

  // Poll stats every 3s while running
  useEffect(() => {
    if (!running) return;
    const id = setInterval(fetchStats, 3000);
    return () => clearInterval(id);
  }, [running, fetchStats]);

  // Connect on mount
  useEffect(() => { connect(); return () => ws.current?.close(); }, [connect]);

  return {
    connected, running, step, history, stats, comparison, loadingComparison,
    start, pause, reset, fetchStats, fetchComparison, ACTION_LABELS,
  };
}
