# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Sistema Autónomo de Aprendizaje por Refuerzo para Optimización Dinámica de Decisiones en Entornos Cambiantes**

A full-stack AI system combining Reinforcement Learning, predicate-logic constraints, ML-based environment prediction, and classic search (A*/Greedy) for comparison — exposed via a FastAPI backend and visualized in a React + TypeScript frontend.

## Stack (security-first choices)

### Frontend
| Tool | Replaces | Why |
|------|----------|-----|
| **Bun** | npm / yarn | Does NOT run `pre/postinstall` scripts of dependencies by default; whitelist model via `trustedDependencies`; `minimumReleaseAge` blocks freshly-published packages (supply chain protection) |
| **Vite** | Create React App | CRA is deprecated; Vite is faster and actively maintained |
| **TypeScript** | plain JS | Catches type errors at compile time |
| **Biome** | ESLint + Prettier | Single Rust-based tool for lint + format; no plugin supply-chain risk |
| **Vitest** | Jest | Native Vite integration; no extra config |

### Backend
| Tool | Replaces | Why |
|------|----------|-----|
| **uv** | pip + virtualenv + pyenv | 10–100x faster; universal lockfile (`uv.lock`) for reproducible, auditable installs; written in Rust |
| **Ruff** | flake8 + pylint + black | Single Rust-based linter/formatter |
| **pyproject.toml** | requirements.txt | Standard PEP 517/518; works natively with uv |
| **Pydantic Settings** | python-dotenv | Type-safe config from env vars; fails fast on missing/wrong-typed values |

**Core libraries:** FastAPI (REST + WebSocket), Pandas, NumPy, PyTorch (DQN), scikit-learn, pytest

## Architecture

```
React + TypeScript (Vite)
    ↓ HTTP / WebSocket
FastAPI Backend
    ↓
Intelligent Engine
├── Environment Simulator   — state space, transitions, reward function, dynamic events
├── RL Agent                — Q-Table (Q-Learning) or DQN with epsilon-greedy + experience replay
├── Logic Engine            — ≥10 predicate-logic rules that filter valid actions per state
├── ML Module               — scikit-learn model predicting environment behavior / event probs
├── Classic Search          — A* and/or Greedy for baseline comparison against RL
└── Pandas Analytics        — episode logging, learning curves, performance metrics
```

**Data flow per step:**
1. Environment produces current state.
2. Logic Engine filters the action space using symbolic rules.
3. ML Module provides predicted next-state probabilities.
4. RL Agent picks an action (explore vs. exploit).
5. Environment transitions, emits reward (`+10` correct, `−5` incorrect, `−10` critical failure).
6. Pandas logs the step; FastAPI pushes update to React over WebSocket.

## Problem Formulation

| Symbol | Definition |
|--------|-----------|
| **S** | Agent position, environment conditions, available resources |
| **A** | Move, allocate resources, wait, react |
| **R** | +10 correct decision / −5 incorrect / −10 critical failure |

RL update rule: `Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') − Q(s,a)]`

## Commands

### Backend (uv)
```bash
uv sync                               # install all deps from uv.lock
uv run uvicorn main:app --reload      # dev server
uv run pytest                         # all tests
uv run pytest tests/test_agent.py -k rl  # single file / keyword
uv run ruff check .                   # lint
uv run ruff format .                  # format
```

### Frontend (Bun)
```bash
bun install                 # install (no dependency scripts run by default)
bun run dev                 # Vite dev server
bun run build               # production build
bun test                    # Vitest
bun run lint                # Biome lint
bun run format              # Biome format
```

## Security Configuration

### `bunfig.toml` (frontend root)
```toml
[install]
minimumReleaseAge = 259200  # block packages published < 3 days ago (supply chain protection)
```
Add packages that legitimately need install scripts (e.g. native binaries) to `trustedDependencies` in `package.json` — never use a blanket override.

### FastAPI
- Configure CORS explicitly (`allow_origins` whitelist, never `"*"` in production).
- Load all secrets via Pydantic `BaseSettings` from environment variables; never hardcode.
- Use HTTPS in any deployed environment.

## Key Deliverables

1. Complete Python + React/TypeScript source code.
2. Functional dynamic-environment simulation.
3. Technical report: formal S/A/R definitions, convergence analysis, RL vs. A* vs. Greedy comparison.

## Specification

The full project specification is in `Proyecto IA.pdf`.
