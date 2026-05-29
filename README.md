# Sistema Autónomo de RL para Optimización Dinámica de Decisiones

**Curso:** Inteligencia Artificial — Proyecto Final  
**Stack:** FastAPI · PyTorch · SWI-Prolog · scikit-learn · React · TypeScript

Sistema de agente inteligente que aprende a gestionar una **red eléctrica inteligente (Smart Grid)** de 5 sectores combinando cuatro paradigmas de IA: aprendizaje por refuerzo, lógica simbólica, machine learning y búsqueda clásica, con visualización en tiempo real.

---

## Demo del sistema

```
┌─────────────────────────────────────────────────┐
│  SMART·GRID  [WS OK] [PROLOG OK] [ML OK]  EP 42 │
├──────────┬───────────────────────────┬───────────┤
│  Left    │   Operaciones / Histórico │   Right   │
│ Sidebar  │   Entrenamiento / Reglas  │   Panel   │
│ Nav      │   Comparación             │ Sparklines│
│ Sectores │                           │ Controles │
│ Stats    │                           │ Auto-save │
└──────────┴───────────────────────────┴───────────┘
```

---

## Características principales

| Componente | Descripción |
|---|---|
| **Q-Learning** | Agente tabular 165×4, convergencia rápida |
| **DQN** | Red neuronal MLP 3→64→64→4 con replay buffer y target network |
| **Prolog (SWI)** | 10 reglas de predicados en `rules.pl`, filtrado O(1) via LUT |
| **Random Forest** | Predicción online del estado siguiente, reentrenamiento cada 50 pasos |
| **A\* y Greedy** | Baseline de comparación contra RL |
| **Persistencia** | Q-tabla, checkpoint DQN, modelo ML y CSV histórico entre sesiones |
| **Dashboard React** | 5 vistas navegables, sparklines en tiempo real, curva multi-agente |

---

## Requisitos

| Herramienta | Versión mínima | Uso |
|---|---|---|
| Python | 3.12 | Backend |
| [uv](https://github.com/astral-sh/uv) | 0.4+ | Gestor de dependencias Python |
| [Bun](https://bun.sh) | 1.3+ | Gestor de dependencias JS |
| [SWI-Prolog](https://www.swi-prolog.org/Download.html) | 10.0 | Motor lógico (opcional — hay fallback Python) |

---

## Instalación y ejecución

### Backend

```bash
cd backend
uv sync                                        # instala todas las dependencias
uv run uvicorn src.main:app --reload           # servidor en http://localhost:8000
```

### Frontend

```bash
cd frontend
bun install                                    # instala dependencias
bun run dev                                    # Vite dev server en http://localhost:5173
```

### Tests

```bash
cd backend
uv run pytest tests/ -q                        # 31 tests (rendimiento + unitarios)
```

---

## Estructura del proyecto

```
Proyecto_IA/
├── backend/
│   ├── src/
│   │   ├── api/
│   │   │   └── routes.py          # REST + WebSocket endpoints
│   │   ├── engine/
│   │   │   ├── environment.py     # Simulador Smart Grid (165 estados, 4 acciones)
│   │   │   ├── rl_agent.py        # Q-Learning tabular
│   │   │   ├── dqn_agent.py       # Deep Q-Network (PyTorch)
│   │   │   ├── logic_engine.py    # Motor Prolog + LUT O(1)
│   │   │   ├── rules.pl           # 10 reglas ISO Prolog
│   │   │   ├── ml_model.py        # RandomForest predictor
│   │   │   ├── search.py          # A* y Greedy
│   │   │   └── analytics.py       # Acumuladores O(1), CSV append
│   │   └── main.py                # FastAPI app + lifespan persistence
│   ├── data/                      # Auto-generado (gitignored, excepto .gitkeep)
│   └── tests/
│       ├── test_engine.py
│       └── test_performance.py    # 31 benchmarks de rendimiento
├── frontend/
│   └── src/
│       ├── App.tsx                # Shell de 4 zonas + router de vistas
│       ├── hooks/
│       │   └── useSimulation.ts   # WebSocket + estado global
│       └── components/
│           ├── LeftSidebar.tsx    # Nav, sectores, estadísticas
│           ├── RightPanel.tsx     # Sparklines, controles, auto-save
│           ├── EnvironmentMap.tsx
│           ├── DecisionAnatomy.tsx
│           ├── EpisodeChart.tsx   # Curva multi-agente (Q-Learning/DQN)
│           ├── RulesPanel.tsx     # 10 reglas Prolog en tiempo real
│           └── ComparisonView.tsx # RL vs A* vs Greedy
├── reporte_tecnico.md             # Reporte formal (S/A/R, convergencia, benchmarks)
└── MANUAL.md                      # Manual técnico completo
```

---

## Formulación del problema

| Símbolo | Definición |
|---|---|
| **S** | `(posición ∈ [0,4], recursos ∈ [0,10], condición ∈ {0,1,2})` — 165 estados |
| **A** | `{mover, asignar_recursos, esperar, reaccionar}` |
| **R** | +10 decisión correcta / −5 incorrecta / −10 fallo crítico |

Regla de actualización Q-Learning: `Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') − Q(s,a)]`

---

## Resultados

| Algoritmo | Reward promedio | Aprende |
|---|---|---|
| Q-Learning (100+ episodios) | 900 – 1 350 | ✅ |
| DQN (100+ episodios) | 800 – 1 500 | ✅ |
| A* (profundidad 3) | 250 – 380 | ✗ |
| Greedy (horizonte 1) | 200 – 350 | ✗ |

Mejor episodio registrado: **1 640** (Q-Learning, sesión de 181 episodios)

---

## Documentación

- **[Reporte Técnico](reporte_tecnico.md)** — formulación formal, análisis de convergencia, benchmarks, comparación de algoritmos
- **[Manual Técnico](MANUAL.md)** — guía de instalación, arquitectura detallada, referencia de API

---

## Optimizaciones de rendimiento destacadas

| Técnica | Mejora |
|---|---|
| LUT precomputada para reglas Prolog (165 estados) | **28× más rápido** |
| Acumuladores incrementales en Analytics | **~4 000× vs Pandas O(n)** |
| Batching WebSocket → React (flush cada 500 ms) | Re-renders: 6.7/s → 2/s |
| Pipeline completa por paso | < 1 ms (headroom ~14 900×) |

---

## Licencia

Proyecto académico — Curso de Inteligencia Artificial.
