# Manual Técnico — Smart Grid RL Dashboard
## Sistema Autónomo de Aprendizaje por Refuerzo para Optimización Dinámica de Decisiones

**Versión:** 0.1.0  
**Fecha:** 2026-05-21  
**Stack:** FastAPI · PyTorch · SWI-Prolog · React · TypeScript · Vite · Bun · uv

---

## Tabla de contenidos

1. [Visión general](#1-visión-general)
2. [Arquitectura del sistema](#2-arquitectura-del-sistema)
3. [Instalación y puesta en marcha](#3-instalación-y-puesta-en-marcha)
4. [Backend — componentes del motor IA](#4-backend--componentes-del-motor-ia)
   - 4.1 Entorno de simulación
   - 4.2 Agente Q-Learning
   - 4.3 Agente DQN
   - 4.4 Motor de reglas Prolog + LUT
   - 4.5 Módulo ML predictivo
   - 4.6 Búsqueda clásica (A* y Greedy)
   - 4.7 Analytics y acumuladores
5. [API REST y WebSocket](#5-api-rest-y-websocket)
6. [Sistema de persistencia](#6-sistema-de-persistencia)
7. [Frontend — interfaz de usuario](#7-frontend--interfaz-de-usuario)
   - 7.1 Layout de 4 zonas
   - 7.2 LeftSidebar
   - 7.3 RightPanel
   - 7.4 Vistas del contenido principal
8. [Optimizaciones de rendimiento](#8-optimizaciones-de-rendimiento)
9. [Flujo de datos por paso](#9-flujo-de-datos-por-paso)
10. [Interpretación de resultados](#10-interpretación-de-resultados)
11. [Referencia de archivos](#11-referencia-de-archivos)

---

## 1. Visión general

El sistema simula la gestión autónoma de una red eléctrica inteligente (Smart Grid) de 5 sectores. Un agente de aprendizaje por refuerzo toma decisiones en tiempo real bajo restricciones de lógica simbólica (Prolog), con predicción del entorno mediante ML y comparación contra algoritmos de búsqueda clásica.

**Problema formal:**

| Símbolo | Definición |
|---------|-----------|
| **S** | `(posición ∈ {0..4}, recursos ∈ {0..10}, condición ∈ {normal, alerta, fallo})` |
| **A** | `{mover, asignar_recursos, esperar, reaccionar}` |
| **R** | `+10` decisión correcta / `−5` incorrecta / `−10` fallo crítico |
| **Q update** | `Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') − Q(s,a)]` |

**Espacio de estados:** 5 × 11 × 3 = **165 estados** discretos  
**Episodio:** termina a los 200 pasos o cuando `recursos ≤ 0`

---

## 2. Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────────┐
│  React + TypeScript  (Vite · Bun)                           │
│  ┌──────────┐  ┌──────────────────────┐  ┌──────────────┐  │
│  │LeftSidebar│  │   Main Content       │  │  RightPanel  │  │
│  │Nav · Stats│  │ (cambia por vista)   │  │Sparklines    │  │
│  │Sectores  │  │ Operaciones          │  │Métricas      │  │
│  │Sistema   │  │ Entrenamiento        │  │Controles     │  │
│  └──────────┘  │ Reglas · Comparación │  │Auto-guardado │  │
│                │ Histórico            │  └──────────────┘  │
│                └──────────────────────┘                     │
└────────────────────────┬───────────────────────────────────┘
            HTTP REST  / WebSocket (/api/ws)
┌────────────────────────▼───────────────────────────────────┐
│  FastAPI Backend                                            │
│                                                             │
│  ┌─────────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │ Environment │  │ RLAgent  │  │ DQNAgent │  │ Logic  │  │
│  │ 165 estados │  │ Q-tabla  │  │ MLP NN   │  │ Engine │  │
│  │ 4 acciones  │  │ 165×4    │  │3→64→64→4 │  │+LUT O1 │  │
│  └─────────────┘  └──────────┘  └──────────┘  └────────┘  │
│  ┌─────────────┐  ┌──────────┐  ┌──────────────────────┐   │
│  │ ML Predictor│  │ Search   │  │ Analytics            │   │
│  │ RandomForest│  │ A* Greedy│  │ Acumuladores O(1)    │   │
│  └─────────────┘  └──────────┘  └──────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  backend/data/   (auto-generado, gitignored)        │   │
│  │  qtable.npy · dqn_checkpoint.pt · ml_model.joblib  │   │
│  │  training_log.csv · episode_rewards.csv            │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

---

## 3. Instalación y puesta en marcha

### Prerequisitos

- Python ≥ 3.11 + [uv](https://docs.astral.sh/uv/)
- Node.js ≥ 18 + [Bun](https://bun.sh/)
- SWI-Prolog *(opcional — el sistema usa fallback Python si no está instalado)*

### Backend

```bash
cd backend
uv sync                               # instala dependencias desde uv.lock
uv run uvicorn main:app --reload      # servidor en http://localhost:8000
```

El servidor carga automáticamente los datos persistidos de sesiones anteriores al arrancar.

### Frontend

```bash
cd frontend
bun install                           # instala sin ejecutar scripts de dependencias
bun run dev                           # servidor Vite en http://localhost:5173
```

El proxy Vite redirige `/api` → `http://localhost:8000` y `/api/ws` → WebSocket, eliminando CORS.

### Comandos de desarrollo

```bash
# Backend
uv run pytest                         # todos los tests (incluye benchmarks)
uv run pytest tests/test_performance.py -v   # solo benchmarks
uv run ruff check .                   # linter
uv run ruff format .                  # formateador

# Frontend
bun run build                         # build de producción
bun test                              # Vitest
bun run lint                          # Biome lint
```

---

## 4. Backend — componentes del motor IA

### 4.1 Entorno de simulación

**Archivo:** `backend/src/engine/environment.py`

Simula una red de 5 sectores con eventos dinámicos estocásticos:

| Evento | Probabilidad | Efecto |
|--------|-------------|--------|
| Fallo crítico | 5% | `condición=2`, recursos `-3` |
| Alerta | 15% | `condición +1` (hasta 2) |
| Recuperación | 20% | `condición -1` (hasta 0) |
| Sin cambio | 60% | — |

**Función de recompensa por acción:**

| Acción | Condición normal | Condición alerta/fallo |
|--------|-----------------|----------------------|
| `mover (0)` | +10 | -5 |
| `asignar (1)` | +10 (si recursos < 10) | -5 (si al máximo) |
| `esperar (2)` | +5 | -5 |
| `reaccionar (3)` | -5 (si no hay fallo) | +10 (reduce condición) |

### 4.2 Agente Q-Learning

**Archivo:** `backend/src/engine/rl_agent.py`

Agente tabular con política ε-greedy:

```
Q-tabla: 165 estados × 4 acciones = 660 valores float64
α (tasa de aprendizaje):  0.1
γ (factor de descuento):  0.95
ε inicial:                1.0  (exploración total)
ε mínimo:                 0.05 (explotación mayoritaria)
ε decay por paso:         × 0.995
```

**Regla de Bellman implementada:**
```
Q(s,a) ← Q(s,a) + 0.1 × [r + 0.95 × max_a' Q(s',a') − Q(s,a)]
```

**Persistencia:** `qtable.npy` (tabla) + `qtable_meta.npy` (epsilon). Al cargar, valida que las dimensiones coincidan con el entorno actual.

### 4.3 Agente DQN

**Archivo:** `backend/src/engine/dqn_agent.py`

Red neuronal con experiencia replay y red objetivo (*target network*):

```
Arquitectura QNetwork:
  Entrada: 3 neuronas (posición/4, recursos/10, condición/2) — normalizado [0,1]
  Capa 1:  Linear(3→64) + ReLU
  Capa 2:  Linear(64→64) + ReLU
  Salida:  Linear(64→4) — un Q-valor por acción

ReplayBuffer: deque circular, capacidad 10 000 transiciones
Batch size:   64
lr (Adam):    1e-3
γ:            0.95
ε decay:      × 0.995 por paso (igual que Q-Learning)
Target sync:  cada 100 pasos de gradiente
Grad clipping: max_norm = 10.0 (previene exploding gradients)
```

**Diferencias clave vs Q-Learning:**
- Requiere llenar el buffer (64 mínimo) antes de entrenar
- La loss (MSE) se expone en el dashboard en tiempo real
- Más volátil en espacios pequeños, más potente en espacios continuos/grandes
- Persistencia: `dqn_checkpoint.pt` con `weights_only=True` (seguro)

### 4.4 Motor de reglas Prolog + LUT O(1)

**Archivo:** `backend/src/engine/logic_engine.py`

El motor implementa **10 reglas de lógica predicativa** que filtran el espacio de acciones antes de que el agente decida:

| Regla | Acción bloqueada | Condición de bloqueo |
|-------|-----------------|---------------------|
| R1 | mover | recursos < 2 |
| R2 | mover | condición = fallo crítico |
| R3 | asignar | recursos ≥ 10 (máximo) |
| R4 | asignar | fallo crítico y recursos < 1 |
| R5 | esperar | condición = fallo crítico |
| R6 | esperar | recursos ≤ 1 y condición ≥ alerta |
| R7 | reaccionar | condición = normal (sin fallo) |
| R8 | reaccionar | recursos < 1 |
| R9 | reaccionar | sector 0 (industrial) sin alerta |
| R10 | reaccionar | recursos > 8 y red estable |

**Backend dual con fallback transparente:**
1. **Prolog (preferido):** carga `rules.pl` via pyswip al importar. Muestra badge "PROLOG OK" en el header.
2. **Python (fallback):** mismas 10 reglas como lambdas. Activo si SWI-Prolog no está instalado.

**Optimización LUT (Lookup Table):**

Como el espacio de estados es finito (165 estados), ambas funciones están precomputadas al iniciar el módulo:

```python
# Construcción en tiempo de carga del módulo (una sola vez)
_FILTER_LUT, _RULES_LUT = _build_lut()  # 165 entradas cada una

# Llamadas en tiempo real — O(1)
filter_actions_fast(state)     # dict lookup por state["index"]
get_active_rules_fast(state)   # dict lookup por state["index"]
```

**Benchmark comparativo:**
| Método | Velocidad | Mejora |
|--------|-----------|--------|
| `filter_actions` (evaluación predicados) | 465 K llamadas/s | base |
| `filter_actions_fast` (LUT) | 13 M llamadas/s | **30× más rápido** |

Siempre retorna al menos una acción (`[2] = esperar`) si todas están bloqueadas.

### 4.5 Módulo ML predictivo

**Archivo:** `backend/src/engine/ml_model.py`

Predice la probabilidad de cada condición ambiental en el siguiente paso:

- **Modelo:** `RandomForestClassifier` de scikit-learn
- **Features de entrada:** `[posición, recursos, condición, acción]`
- **Target:** condición en el paso t+1
- **Entrenamiento:** automático cada 50 pasos (una vez que hay ≥ 20 registros)
- **Persistencia:** `ml_model.joblib`
- **Exposición:** probabilidades por condición visible en "Decisión Anatómica" del frontend

### 4.6 Búsqueda clásica (A* y Greedy)

**Archivo:** `backend/src/engine/search.py`

Implementa dos algoritmos de referencia para comparar contra el agente RL:

- **Greedy:** selecciona la acción con mayor reward inmediato estimado
- **A*:** búsqueda informada con heurística basada en recursos y condición actual

Se evalúan sobre 10 episodios frescos (entorno reseteado) y el resultado se expone en el endpoint `/simulation/comparison`.

### 4.7 Analytics y acumuladores

**Archivo:** `backend/src/engine/analytics.py`

**Problema resuelto:** `summary()` se llama cada 3 s durante la simulación. La implementación original reconstruía un DataFrame Pandas completo en cada llamada — O(n) creciente. Después de 10 000+ pasos, esto era medible.

**Solución: acumuladores incrementales** que mantienen `summary()` en O(1):

```python
_step_count: int          # contador total de pasos
_resource_sum: float      # suma acumulada de recursos (para promedio)
_action_counts: dict      # histograma de acciones
_condition_counts: dict   # histograma de condiciones
```

Cada llamada a `log()` actualiza los acumuladores en O(1). `summary()` sólo hace divisiones simples.

**Benchmark:**
| Implementación | Latencia a 10 000 pasos | Latencia a 1 pasos |
|---------------|------------------------|-------------------|
| Pandas DataFrame | ~8 ms | ~0.5 ms |
| Acumuladores | **0.002 ms** | **0.002 ms** |

**Cap de memoria:** después de cada flush CSV, `_records` se recorta a los últimos 1 000 registros. Los datos más antiguos están en disco (`training_log.csv`).

---

## 5. API REST y WebSocket

**Prefijo:** `/api`  
**Documentación interactiva:** `http://localhost:8000/docs`

### Endpoints REST

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/status` | Estado del sistema (agente activo, ML, Prolog) |
| `GET` | `/simulation/stats` | Estadísticas de sesión (pasos, episodios, rewards, último guardado) |
| `POST` | `/simulation/reset` | Reset de sesión (preserva CSV histórico) |
| `POST` | `/simulation/reset-full` | Reset total — borra todos los archivos persistidos |
| `POST` | `/simulation/save` | Guardado manual inmediato |
| `POST` | `/simulation/agent-type` | Cambio de agente (`{"type": "qlearning"\|"dqn"}`) |
| `POST` | `/simulation/autosave-interval` | Configura auto-guardado (`{"every": 1..20}`) |
| `GET` | `/simulation/comparison` | Compara RL vs A* vs Greedy (10 episodios c/u) |
| `GET` | `/training/history` | Últimos 300 episodios del CSV histórico |

### WebSocket `/api/ws`

**Protocolo:** el cliente envía `{"action": "step"}` y recibe un JSON por paso:

```json
{
  "state":          { "position": 2, "resources": 7, "env_condition": 0, "index": 63 },
  "action":         1,
  "allowed_actions": [0, 1, 2],
  "reward":         10.0,
  "done":           false,
  "episode":        42,
  "total_reward":   150.0,
  "epsilon":        0.729,
  "ml_probs":       { "0": 0.7, "1": 0.2, "2": 0.1 },
  "ml_trained":     true,
  "agent_type":     "qlearning",
  "logic_engine":   "prolog",
  "decision_mode":  "explotación",
  "q_values":       [2.3, 5.1, -1.2, 0.8],   // cada 5 pasos
  "rules_info":     [...]                       // cada 5 pasos
}
```

**Optimización de payload:** `q_values` y `rules_info` se envían solo cada 5 pasos (reducción ~60% del tamaño del mensaje). El frontend fusiona con el último valor conocido para no perder datos.

**Cadencia:** el frontend envía un step cada 150 ms → ~6.7 pasos/s. La pipeline completa tarda < 1 ms, dejando un margen de headroom de ~14 900×.

---

## 6. Sistema de persistencia

### Archivos generados (en `backend/data/`)

| Archivo | Contenido | Formato |
|---------|-----------|---------|
| `qtable.npy` | Tabla Q 165×4 | NumPy binary |
| `qtable_meta.npy` | Epsilon actual del Q-Learning | NumPy binary |
| `dqn_checkpoint.pt` | Pesos online/target, optimizer, epsilon, step_count | PyTorch |
| `ml_model.joblib` | RandomForest + LabelEncoder entrenados | joblib |
| `training_log.csv` | Registro por paso: episode, position, resources, condition, action, reward | CSV append |
| `episode_rewards.csv` | Registro por episodio: episode_num, total_reward, steps, epsilon, agent_type | CSV append |

> `backend/data/` está en `.gitignore` — los datos de entrenamiento nunca se suben al repositorio.

### Auto-guardado

Configurable desde el panel derecho (1 a 20 episodios). Por defecto: cada episodio.

```python
# En el handler WebSocket, tras cada episodio completado:
if _analytics.episode - _last_saved_episode >= _AUTO_SAVE_EVERY:
    _do_save()
    _last_saved_episode = _analytics.episode
```

### Carga automática al arrancar (lifespan)

```python
@asynccontextmanager
async def lifespan(app):
    _agent_ql.load(DATA_DIR / "qtable.npy")       # restaura Q-tabla y epsilon
    _agent_dqn.load(DATA_DIR / "dqn_checkpoint.pt") # restaura red neuronal
    _predictor.load(DATA_DIR / "ml_model.joblib")  # restaura ML model
    historical = _analytics.load_csv(DATA_DIR)     # carga historial de episodios
    if historical:
        _analytics._episode_rewards = historical
        _analytics.episode = historical[-1]["episode_num"] + 1
    yield
    _do_save()   # flush al apagar
```

### Comportamiento por tipo de reset

| Operación | Q-tabla | DQN | ML | CSV histórico |
|-----------|---------|-----|----|---------------|
| `POST /simulation/reset` | ✓ limpia | ✓ limpia | — | **Preservado** |
| `POST /simulation/agent-type` | — | — | — | **Preservado** |
| `POST /simulation/reset-full` | ✓ limpia | ✓ limpia | ✓ limpia | **Eliminado** |

### Cambio de agente (comportamiento específico)

Al cambiar entre Q-Learning y DQN:
1. Se guarda el estado actual antes de cambiar (no se pierde historial)
2. Se resetea **solo el agente nuevo** (el anterior conserva su Q-tabla/pesos)
3. El entorno se resetea para iniciar un episodio limpio
4. Las estadísticas (`analytics`) continúan acumulando — la curva de aprendizaje es continua

---

## 7. Frontend — interfaz de usuario

### 7.1 Layout de 4 zonas

```css
.app-shell {
  display: grid;
  grid-template-columns: 200px 1fr 280px;
  grid-template-rows: 42px 1fr;
  height: 100vh;
  overflow: hidden;
}
/* Header ocupa las 3 columnas (grid-column: 1 / -1) */
```

```
┌─────────────────────────────────────────────────┐
│  TOP HEADER  42px  —  full width                 │
│  Logo · badges WS/Prolog/ML · EP · SYS · ε      │
├──────────┬───────────────────────────┬───────────┤
│  LEFT    │       MAIN CONTENT        │   RIGHT   │
│ SIDEBAR  │  (cambia con la nav)      │   PANEL   │
│  200px   │                           │   280px   │
│ Nav      │                           │ Métricas  │
│ Sectores │                           │ Sparklines│
│ Eventos  │                           │ Controles │
│ Stats    │                           │ AutoSave  │
│ Sistema  │                           │ Agente RL │
└──────────┴───────────────────────────┴───────────┘
```

### 7.2 LeftSidebar

**Archivo:** `frontend/src/components/LeftSidebar.tsx`

| Sección | Contenido |
|---------|-----------|
| **Brand** | "SMART·GRID" + versión |
| **Navegación** | 5 botones de vista con indicador activo (borde izquierdo azul) |
| **Red — 5 sectores** | S0-S4 con badge de abreviatura, barra de capacidad MW y resaltado del sector activo |
| **Dinámica activa** | Estado actual del entorno (normal/alerta/fallo) con animación de parpadeo en alertas |
| **Estadísticas** | Pasos totales · Episodios · Media últimos 10 · Capacidad promedio |
| **Sistema** | Política activa · Episodio · Estado Prolog · Estado ML · Estado WS |

Las estadísticas se actualizan automáticamente cada 3 segundos durante la simulación.

### 7.3 RightPanel

**Archivo:** `frontend/src/components/RightPanel.tsx`

| Sección | Contenido |
|---------|-----------|
| **Métricas clave** | 4 tarjetas 2×2: Tasa éxito % · Reward avg · Violaciones Prolog · Epsilon ε |
| **Exploración ε** | Barra de progreso visual del epsilon actual |
| **Entrenamiento en vivo** | Badge "X pasos"/"Sin datos" + 4 SparkCards con área rellena |
| **DQN Telemetría** | Loss MSE · Buffer size *(solo si agente = DQN)* |
| **Agente RL** | Selector Q-Learn / DQN |
| **Auto-guardado** | Último guardado · Episodios sin guardar · Selector intervalo · Botón pausar y guardar |
| **Control** | ▶ Iniciar · ⏸ Pausar · ↺ Reset |

**SparkCards — métricas en tiempo real** (ventana últimos 60 pasos):

| SparkCard | Fuente de datos | Color |
|-----------|----------------|-------|
| Reward / paso | `step.reward` (+10/-5/-10) | Verde |
| Tasa éxito % | Rolling 10-step: `reward ≥ 10` | Cyan |
| Actividad simbólica | Rolling 10-step: acciones bloqueadas por Prolog | Ámbar |
| Capacidad MW | `state.resources` (0-10) | Púrpura |

Las sparklines se calculan con `useMemo([history])` — solo recalculan cuando llegan nuevos datos del batch de 500 ms. Estado vacío muestra línea punteada visible con puntos indicadores.

**Auto-guardado — UI:**
- Indicador del último episodio guardado
- Advertencia naranja cuando hay ≥ 5 episodios sin guardar
- Selector de intervalo con actualización optimista (no hay flickering mientras la API responde)
- Botón "Pausar y guardar" — detiene la simulación y guarda en una sola acción

### 7.4 Vistas del contenido principal

#### Operaciones (vista por defecto)

- **EnvironmentMap:** mapa visual de los 5 sectores con estado de condición y posición del agente
- **DecisionAnatomy:** descomposición de la última decisión (estado → reglas Prolog activas → Q-valores → acción elegida → modo exploración/explotación)
- **DecisionLog:** consola con historial de los últimos pasos (reward codificado por color)

#### Entrenamiento

- **AgentView:** Q-valores actuales por acción en gráfico de barras
- **PolicyView:** visualización de la política aprendida
- **Selector de agente:** botones Q-Learning / DQN con descripción del algoritmo activo y telemetría DQN si aplica

#### Reglas Prolog

- **RulesPanel:** las 10 reglas con estado activo/inactivo en el estado actual y el backend utilizado (Prolog/Python)

#### Comparación

- **ComparisonView:** tabla comparativa RL vs A* vs Greedy. Ejecuta 10 episodios de cada algoritmo en entornos frescos y muestra reward promedio

#### Histórico

- **RewardChart:** gráfico de recompensas por paso de la sesión actual
- **EpisodeChart:** curva de aprendizaje multi-agente con:
  - Línea cyan para episodios Q-Learning
  - Línea púrpura para episodios DQN
  - Media móvil de 5 episodios (naranja)
  - Marcadores verticales en cada cambio de agente ("Q-Learning → DQN")
  - Datos históricos de CSV (sesiones anteriores) + sesión actual mezclados
- **Estadísticas de sesión:** pasos · episodios · media últimos 10 · capacidad promedio + botón actualizar

---

## 8. Optimizaciones de rendimiento

### 8.1 Tabla de optimizaciones implementadas

| Componente | Problema | Solución | Mejora |
|------------|---------|----------|--------|
| `filter_actions` | Evaluación predicados O(n) por paso | LUT precomputada para 165 estados | **30× más rápido** |
| `Analytics.summary()` | DataFrame Pandas O(n) cada 3 s | Acumuladores incrementales O(1) | **4 000× más rápido** |
| `_records` en memoria | Crecimiento ilimitado → OOM | Cap en 1 000 entradas post-flush | **Memoria acotada** |
| WS payload | `q_values` + `rules_info` en cada paso | Envío cada 5 pasos; frontend fusiona | **~60% menos payload** |
| React re-renders | `setHistory` en cada mensaje WS (~6.7/s) | Batch flush cada 500 ms | **Re-renders de 6.7/s → 2/s** |
| Bundle JS | Un chunk de 651 KB (Recharts) | `manualChunks` en Vite | **244 KB + 405 KB separados** |
| `EpisodeChart` | Re-render en cada actualización de `step` | `React.memo` | Re-render solo al cambiar `stats` |
| Selects controlados | Flickering al actualizar intervalo | Estado local optimista + sync con server | **Sin flickering** |

### 8.2 Batching de historial

```typescript
// En useSimulation.ts — el WebSocket push acumula en un ref:
socket.onmessage = (e) => {
  const data = mergeWithLastStep(JSON.parse(e.data));
  setStep(data);                        // UI actual: inmediato
  historyBatchRef.current.push(data);   // historial: diferido
};

// Flush al estado React cada 500 ms:
historyFlushRef.current = setInterval(() => {
  const batch = historyBatchRef.current;
  if (batch.length === 0) return;
  historyBatchRef.current = [];
  setHistory(h => {
    const combined = [...h, ...batch];
    return combined.length > 200 ? combined.slice(-200) : combined;
  });
}, 500);
```

### 8.3 Tests de rendimiento

**Archivo:** `backend/tests/test_performance.py` — 31 pruebas en 5 clases:

```
TestFilterActions       — LUT speedup ≥ 3×, correctness vs evaluación directa
TestGetActiveRules      — LUT speedup ≥ 3×
TestAnalyticsSummary    — latencia ≤ 1 ms para summary()
TestRLAgentThroughput   — Q-Learning ≥ 100 000 pasos/s
TestFullPipeline        — headroom WS ≥ 50× (actual: ~14 900×)
```

---

## 9. Flujo de datos por paso

```
Cliente WS → {"action": "step"}
                │
                ▼
1. Logic Engine   filter_actions_fast(state)     → O(1) LUT
2. ML Predictor   predict_proba(state, action)   → probabilidades condición t+1
3. Agent          select_action(state, allowed)  → ε-greedy dentro de acciones permitidas
4. Environment    step(action)                   → (next_state, reward, done)
5. Agent          learn(s, a, r, s', done)       → actualiza Q-tabla o red neuronal
6. Analytics      log(state, action, reward)     → acumuladores O(1)
7. ML retraining  if step_count % 50 == 0        → entrena RandomForest
8. Heavy fields   if step_count % 5 == 0         → agrega q_values, rules_info al mensaje
9. WS send        await websocket.send_json(msg) → ≤ 1 ms total

Si done:
10. env.reset()
11. analytics.next_episode(epsilon, agent_type)
12. if episodes_since_save >= AUTO_SAVE_EVERY: _do_save()
```

---

## 10. Interpretación de resultados

### Curva de aprendizaje

| Patrón | Significado |
|--------|-------------|
| Reward creciente estable | Agente convergiendo correctamente |
| Pico inicial + caída | Exploración aleatoria afortunada → política parcial inestable |
| Valle profundo tras cambio de agente | DQN llenando replay buffer; comportamiento esperado |
| Línea plana en reward bajo | Agente atascado en mínimo local; aumentar ε o resetear |
| Media móvil suavizada ascendente | Convergencia real — el ruido es normal |

### Métricas del panel

| Métrica | Rango saludable | Significado |
|---------|----------------|-------------|
| Tasa éxito % | ≥ 60% | Mayoría de pasos con reward ≥ +10 |
| Epsilon ε | 0.05 – 0.3 | Agente en fase de explotación con algo de exploración |
| Loss DQN | < 1.0 | Red convergiendo; 0.0 con buffer pequeño = no entrena aún |
| Violaciones Prolog | Cualquiera | Las reglas están activas; número alto = estado peligroso frecuente |
| Capacidad MW | ≥ 5 | Recursos suficientes; < 2 activa restricciones de Prolog |

### Comparación RL vs Búsqueda clásica

- **Q-Learning supera a Greedy** típicamente después de 50-100 episodios de entrenamiento
- **DQN puede superar a Q-Learning** a largo plazo en sesiones de > 500 episodios
- **A*** provee el techo teórico — si el RL lo iguala, ha convergido óptimamente

---

## 11. Referencia de archivos

```
Proyecto_IA/
├── CLAUDE.md                         # Instrucciones para Claude Code
├── MANUAL.md                         # Este documento
├── Proyecto IA.pdf                   # Especificación original
│
├── backend/
│   ├── pyproject.toml                # Dependencias + config uv
│   ├── uv.lock                       # Lockfile reproducible
│   ├── data/                         # Auto-generado (gitignored)
│   │   ├── qtable.npy
│   │   ├── qtable_meta.npy
│   │   ├── dqn_checkpoint.pt
│   │   ├── ml_model.joblib
│   │   ├── training_log.csv
│   │   └── episode_rewards.csv
│   ├── src/
│   │   ├── main.py                   # FastAPI app + lifespan (auto-load/save)
│   │   ├── config.py                 # Pydantic Settings (CORS, etc.)
│   │   ├── api/
│   │   │   └── routes.py             # REST endpoints + WebSocket handler
│   │   └── engine/
│   │       ├── environment.py        # Simulador de red eléctrica
│   │       ├── rl_agent.py           # Q-Learning tabular
│   │       ├── dqn_agent.py          # DQN con PyTorch
│   │       ├── logic_engine.py       # Reglas Prolog + LUT O(1)
│   │       ├── rules.pl              # Reglas ISO Prolog (10 predicados)
│   │       ├── ml_model.py           # RandomForest predictor
│   │       ├── search.py             # A* y Greedy
│   │       └── analytics.py          # Logging + acumuladores + CSV
│   └── tests/
│       ├── test_agent.py             # Tests unitarios agentes
│       ├── test_environment.py       # Tests entorno
│       ├── test_logic_engine.py      # Tests reglas + LUT
│       └── test_performance.py       # 31 benchmarks de rendimiento
│
└── frontend/
    ├── package.json
    ├── bunfig.toml                   # Seguridad: minimumReleaseAge = 3 días
    ├── vite.config.ts                # Proxy API + manualChunks (recharts separado)
    ├── biome.json                    # Linter + formatter
    └── src/
        ├── App.tsx                   # Shell 4 zonas + view switching + mergedEpisodes
        ├── index.css                 # Sistema CSS completo (grid, componentes)
        ├── hooks/
        │   └── useSimulation.ts      # WS + estado + batching + persistencia
        └── components/
            ├── LeftSidebar.tsx       # Nav · Sectores · Stats · Sistema
            ├── RightPanel.tsx        # Métricas · Sparklines · Controles · AutoSave
            ├── StartupModal.tsx      # Modal inicial: continuar vs reiniciar
            ├── EnvironmentMap.tsx    # Mapa visual de la red
            ├── DecisionAnatomy.tsx   # Descomposición de decisión
            ├── DecisionLog.tsx       # Consola de pasos recientes
            ├── AgentView.tsx         # Q-valores actuales
            ├── PolicyView.tsx        # Política aprendida
            ├── RulesPanel.tsx        # 10 reglas Prolog con estado activo
            ├── ComparisonView.tsx    # RL vs A* vs Greedy
            ├── RewardChart.tsx       # Rewards por paso (React.memo)
            ├── EpisodeChart.tsx      # Curva aprendizaje multi-agente (React.memo)
            └── StartupModal.tsx      # Selector continuar/resetear al inicio
```
