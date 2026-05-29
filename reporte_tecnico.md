# Reporte Técnico
## Sistema Autónomo de Aprendizaje por Refuerzo para Optimización Dinámica de Decisiones en Entornos Cambiantes

**Curso:** Inteligencia Artificial  
**Fecha:** Mayo 2026  
**Stack:** FastAPI · PyTorch · SWI-Prolog · scikit-learn · React · TypeScript · Vite

---

## Repositorio del proyecto

| | |
|---|---|
| **GitHub** | https://github.com/Anthony8-11/proyecto-ai |
| **Rama principal** | `main` / `develop` |
| **Instrucciones de instalación** | Ver `README.md` en el repositorio |

---

## Tabla de contenidos

1. [Introducción](#1-introducción)
2. [Formulación formal del problema](#2-formulación-formal-del-problema)
3. [Arquitectura del sistema](#3-arquitectura-del-sistema)
4. [Aprendizaje por refuerzo](#4-aprendizaje-por-refuerzo)
5. [Motor de lógica simbólica con Prolog](#5-motor-de-lógica-simbólica-con-prolog)
6. [Módulo de predicción ML](#6-módulo-de-predicción-ml)
7. [Búsqueda clásica (A* y Greedy)](#7-búsqueda-clásica-a-y-greedy)
8. [Sistema de persistencia y sesiones continuas](#8-sistema-de-persistencia-y-sesiones-continuas)
9. [Optimizaciones de rendimiento](#9-optimizaciones-de-rendimiento)
10. [Dashboard de visualización](#10-dashboard-de-visualización)
11. [Análisis de convergencia](#11-análisis-de-convergencia)
12. [Resultados comparativos](#12-resultados-comparativos)
13. [Stack tecnológico](#13-stack-tecnológico)
14. [Decisiones de diseño](#14-decisiones-de-diseño)
15. [Conclusiones](#15-conclusiones)

---

## 1. Introducción

Este proyecto implementa un sistema de agente inteligente que aprende a tomar decisiones óptimas en un entorno dinámico y parcialmente impredecible, modelando la gestión autónoma de una red eléctrica inteligente (Smart Grid) de 5 sectores. El agente combina cuatro paradigmas de IA:

- **Aprendizaje por Refuerzo** (Q-Learning y DQN) como mecanismo principal de decisión
- **Motor de lógica simbólica** con 10 reglas de predicados (SWI-Prolog) que restringen el espacio de acciones
- **Modelo de ML** (Random Forest) que predice el comportamiento futuro del entorno
- **Búsqueda clásica** (A* y Greedy) como baseline de comparación

El sistema incluye persistencia de entrenamiento entre sesiones, visualización en tiempo real mediante WebSocket, y un dashboard interactivo con métricas, curvas de aprendizaje multi-agente y telemetría de DQN.

---

## 2. Formulación Formal del Problema

### 2.1 Espacio de estados S

Cada estado del sistema queda definido por una tupla de tres variables:

```
s = (posición, recursos, condición_entorno)
```

| Variable | Rango | Descripción |
|---|---|---|
| `posición` | 0 – 4 | Sector activo del agente en la red (0=Industrial, 1=Res. Norte, 2=Comercial, 3=Res. Sur, 4=Planta) |
| `recursos` | 0 – 10 | Capacidad disponible en MW |
| `condición_entorno` | 0, 1, 2 | 0 = normal · 1 = alerta (sobrecarga) · 2 = crítico (fallo de red) |

El espacio total de estados es `|S| = 5 × 11 × 3 = 165 estados discretos`.

Adicionalmente, el estado incluye un campo `index` que codifica la tupla en un entero único:

```
index = posición × 33 + recursos × 3 + condición
```

Este índice se usa para el acceso O(1) a las tablas precomputadas del motor de reglas y la Q-tabla.

### 2.2 Espacio de acciones A

El agente dispone de cuatro acciones discretas:

| Índice | Acción | Efecto en el entorno |
|---|---|---|
| 0 | `mover` | Avanza al siguiente sector: `(pos + 1) mod 5` |
| 1 | `asignar_recursos` | Incrementa capacidad en +2 MW (máx. 10) |
| 2 | `esperar` | Mantiene estado actual; monitoreo pasivo |
| 3 | `reaccionar` | Reduce condición en −1, consume 1 MW de capacidad |

### 2.3 Función de recompensa R

```
R(s, a) =  +10   si la acción es apropiada para el contexto actual
            −5   si la acción es incorrecta o innecesaria
           −10   si ocurre un fallo crítico (recursos = 0)
```

Reglas de recompensa concretas:

| Acción | Condición del entorno | Recompensa |
|---|---|---|
| `mover` | normal (0) | +10 |
| `mover` | alerta (1) o crítico (2) | −5 |
| `asignar_recursos` | recursos < 10 | +10 |
| `asignar_recursos` | recursos ya en 10 (máximo) | −5 |
| `esperar` | normal (0) | +5 |
| `esperar` | alerta (1) o crítico (2) | −5 |
| `reaccionar` | alerta (1) o crítico (2) | +10 |
| `reaccionar` | normal (0) — sin amenaza activa | −5 |

### 2.4 Dinámica del entorno

En cada paso, antes de aplicar la acción del agente, ocurre un evento aleatorio estocástico:

```
P(evento crítico)    = 0.05  →  condición = 2, recursos − 3
P(evento de alerta)  = 0.15  →  condición + 1 (máx. 2)
P(recuperación)      = 0.20  →  condición − 1 (mín. 0)
P(sin cambio)        = 0.60  →  estado sin modificar
```

Un episodio termina cuando el agente acumula **200 pasos** o cuando `recursos ≤ 0`.

---

## 3. Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│  React + TypeScript  (Vite · Bun)                           │
│  ┌──────────┐  ┌──────────────────────┐  ┌──────────────┐  │
│  │LeftSidebar│  │   Main Content       │  │  RightPanel  │  │
│  │Nav · Stats│  │ Operaciones          │  │ Métricas     │  │
│  │Sectores  │  │ Entrenamiento        │  │ Sparklines   │  │
│  │Sistema   │  │ Reglas · Comparación │  │ Controles    │  │
│  └──────────┘  │ Histórico            │  │ Auto-guardado│  │
│                └──────────────────────┘  └──────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
              HTTP REST / WebSocket (/api/ws)
┌─────────────────────────▼───────────────────────────────────┐
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
│  backend/data/  (auto-generado, persistente entre sesiones) │
│  qtable.npy · dqn_checkpoint.pt · episode_rewards.csv      │
└────────────────────────────────────────────────────────────┘
```

**Flujo por paso (pipeline completa):**

1. El entorno aplica un evento dinámico aleatorio y devuelve el estado `s`.
2. El motor de lógica consulta la LUT precomputada: O(1) → acciones permitidas.
3. El predictor ML estima la probabilidad de cada condición futura.
4. El agente RL selecciona acción `a` (ε-greedy) dentro de las acciones permitidas.
5. El entorno transiciona a `s'`, emite recompensa `r` y flag `done`.
6. El agente actualiza su Q-tabla o red neuronal con la transición `(s, a, r, s', done)`.
7. Analytics registra el paso en O(1) con acumuladores incrementales.
8. Cada 50 pasos: reentrenamiento del predictor ML con los últimos registros.
9. Cada 5 pasos: se incluyen `q_values` y `rules_info` en el mensaje WebSocket.
10. FastAPI envía la actualización al frontend; tiempo total de pipeline < 1 ms.

---

## 4. Aprendizaje por Refuerzo

### 4.1 Q-Learning tabular

**Regla de actualización (Bellman):**

```
Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]
```

| Hiperparámetro | Valor | Justificación |
|---|---|---|
| α (tasa de aprendizaje) | 0.1 | Balance entre estabilidad y velocidad de actualización |
| γ (factor de descuento) | 0.95 | Valoración alta de recompensas futuras en horizonte de 200 pasos |
| ε inicial | 1.0 | Exploración total al inicio para construir la Q-tabla |
| ε mínimo | 0.05 | Exploración residual permanente para adaptarse a cambios |
| decay de ε | × 0.995 por paso | Decaimiento exponencial; ε = 0.05 se alcanza ≈ paso 600 |

La tabla Q tiene dimensiones `165 × 4 = 660 valores float64`. Se inicializa en cero. La política es ε-greedy: con probabilidad `ε` se elige acción aleatoria entre las permitidas; con `1−ε` se elige `argmax_a Q(s,a)` restringido a las acciones que Prolog no ha bloqueado.

**Trayectoria de ε durante el entrenamiento:**

```
ε(t) = max(0.05, 1.0 × 0.995^t)

t = 100 pasos:  ε ≈ 0.607   (aún explorando mayoritariamente)
t = 300 pasos:  ε ≈ 0.223   (balance explotación/exploración)
t = 460 pasos:  ε ≈ 0.100   (explotación dominante)
t = 600 pasos:  ε = 0.050   (mínimo alcanzado y mantenido)
```

**Persistencia:** la Q-tabla se guarda en `qtable.npy` (NumPy binary) y el ε actual en `qtable_meta.npy`. Al reiniciar el backend, ambos se restauran automáticamente — el agente continúa exactamente donde dejó.

### 4.2 Deep Q-Network (DQN)

En lugar de almacenar Q-values en una tabla, una red neuronal aproxima la función `Q(s,a;θ)`.

**Arquitectura QNetwork (MLP):**

```
Entrada:  [posición/4,  recursos/10,  condición/2]  → vector normalizado en [0,1]
Capa 1:   Linear(3 → 64) + ReLU
Capa 2:   Linear(64 → 64) + ReLU
Salida:   Linear(64 → 4)  → un Q-value por acción
```

**Componentes adicionales del algoritmo DQN:**

| Componente | Configuración | Propósito |
|---|---|---|
| Replay Buffer | Deque circular, cap 10 000 transiciones | Rompe correlación temporal entre muestras |
| Batch size | 64 | Mínimo para gradientes informativos |
| Target Network | Copia congelada, sincronización cada 100 pasos | Estabiliza los targets de Bellman |
| Función de pérdida | MSE | `L = (Q_online(s,a) − y)²` |
| Optimizador | Adam, lr = 1×10⁻³ | Adaptativo; robusto a diferentes escalas |
| Gradient clipping | max_norm = 10.0 | Previene *exploding gradients* en episodios iniciales |
| ε decay | × 0.995 por paso | Igual que Q-Learning para comparabilidad |

**Target de Bellman con red objetivo:**

```
y = r                                      si done = True
y = r + γ · max_a' Q_target(s', a'; θ⁻)  si done = False
```

El DQN comienza a entrenar cuando el replay buffer tiene ≥ 64 transiciones. Con menos datos, `learn()` retorna inmediatamente (no hay gradiente). Esto explica el periodo de inestabilidad visible en la curva de aprendizaje justo tras cambiar a DQN.

**Persistencia DQN:** checkpoint completo en `dqn_checkpoint.pt` con `weights_only=True` (seguro). Incluye pesos online/target, estado del optimizador, ε actual y contador de pasos de gradiente.

### 4.3 Cambio de agente en tiempo de ejecución

Ambos agentes coexisten en memoria durante toda la sesión. Al cambiar de agente:

1. Se guarda el estado actual antes de cambiar (sin pérdida de datos).
2. Se resetea **solo el agente nuevo** — el anterior conserva su Q-tabla o pesos de red.
3. El entorno se resetea para iniciar un episodio limpio.
4. Las estadísticas acumuladas continúan — la curva de aprendizaje es ininterrumpida.
5. Cada episodio queda etiquetado con `agent_type` en el CSV histórico.

---

## 5. Motor de Lógica Simbólica con Prolog

El motor de restricciones está implementado en **SWI-Prolog** (v10.0.2) e integrado a Python mediante la biblioteca `pyswip`. Las reglas se escriben en sintaxis Prolog ISO en `backend/src/engine/rules.pl` y se cargan una única vez al arrancar el backend.

### 5.1 Arquitectura de integración

```
Python (WebSocket handler)
    │  filter_actions_fast(state)       ← O(1) LUT lookup
    ▼
logic_engine.py — LUT precomputada
    │  (equivalente a consultar Prolog para todos los 165 estados)
    ▼
Lista de acciones permitidas → agente RL

Para visualización (cada 5 pasos):
    get_active_rules_fast(state)        ← O(1) LUT lookup
    → lista de 10 reglas con flag active=True/False
```

El módulo tiene **fallback automático**: si SWI-Prolog no está disponible, `logic_engine.py` utiliza predicados Python equivalentes. El sistema continúa sin cambios en el código cliente.

### 5.2 Reglas en sintaxis Prolog (rules.pl)

Las constantes del dominio se declaran como hechos Prolog:

```prolog
min_recursos(2).
critico(2).
alerta(1).
```

Las 10 reglas de prohibición se expresan como cláusulas Horn. Algunos ejemplos representativos:

```prolog
% R1: No moverse sin recursos mínimos
no_accion(mover, _Pos, Rec, _Cond) :-
    min_recursos(Min), Rec < Min.

% R2: No moverse en condición crítica
no_accion(mover, _Pos, _Rec, Cond) :-
    critico(Crit), Cond =:= Crit.

% R5: Esperar en condición crítica es inválido
no_accion(esperar, _Pos, _Rec, Cond) :-
    critico(Crit), Cond =:= Crit.

% R7: Reaccionar sin amenaza activa es inútil
no_accion(reaccionar, _Pos, _Rec, 0).

% R9: No cortar al sector industrial sin alerta
no_accion(reaccionar, 0, _Rec, Cond) :-
    alerta(Alert), Cond < Alert.
```

La regla de permisión usa **negación por fallo** (`\+`), mecanismo central de Prolog para razonamiento por defecto:

```prolog
accion_permitida(Accion, Pos, Rec, Cond) :-
    \+ no_accion(Accion, Pos, Rec, Cond).
```

### 5.3 Tabla de las 10 reglas

| Regla | Acción bloqueada | Condición de bloqueo |
|---|---|---|
| R1 | mover | recursos < 2 |
| R2 | mover | condición == crítico (2) |
| R3 | asignar | recursos == 10 (máximo) |
| R4 | asignar | condición == crítico AND recursos < 1 |
| R5 | esperar | condición == crítico (2) |
| R6 | esperar | recursos ≤ 1 AND condición ≥ alerta |
| R7 | reaccionar | condición == normal (0) |
| R8 | reaccionar | recursos < 1 |
| R9 | reaccionar | posición == 0 (industrial) AND condición < alerta |
| R10 | reaccionar | recursos > 8 AND condición == normal |

Si todas las acciones resultan prohibidas, el sistema retorna `[2]` (esperar) como garantía de acción mínima siempre disponible.

### 5.4 Optimización LUT — O(1) en tiempo de ejecución

El espacio de estados es finito (165 estados). Como `filter_actions` y `get_active_rules` son funciones puras del estado, sus resultados se precomputan **una sola vez al cargar el módulo** y se sirven desde un diccionario:

```python
# Tiempo de carga: ~0.3 ms para generar las 165 entradas
_FILTER_LUT, _RULES_LUT = _build_lut()

# Tiempo de ejecución por paso: O(1) dict lookup
filter_actions_fast(state)    # = _FILTER_LUT[state["index"]]
get_active_rules_fast(state)  # = _RULES_LUT[state["index"]]
```

| Método | Velocidad | Factor de mejora |
|---|---|---|
| `filter_actions` con evaluación de predicados | 465 000 llamadas/s | base |
| `filter_actions_fast` con LUT | 13 000 000 llamadas/s | **28×** |

---

## 6. Módulo de Predicción ML

**Modelo:** `RandomForestClassifier` de scikit-learn (50 estimadores, criterio Gini).

**Objetivo:** predecir la condición del entorno en el próximo paso dado el estado y la acción actual, dando al agente capacidad de anticipación.

**Variables:**

```
X = [posición, recursos, condición_actual, acción]   → features de entrada
y = condición_siguiente (0, 1 o 2)                  → variable objetivo
```

**Entrenamiento online progresivo:**

- El modelo se re-entrena cada **50 pasos** usando todos los registros acumulados hasta ese momento.
- Requiere un mínimo de **20 registros** para la primera inferencia.
- Hasta alcanzar ese mínimo, retorna distribución uniforme `{0: 0.33, 1: 0.33, 2: 0.33}`.
- La función `predict_proba(state, action)` devuelve probabilidades por clase, visibles en el panel "Decisión Anatómica" del dashboard.

**Persistencia:** el modelo entrenado se serializa con `joblib` en `ml_model.joblib` y se restaura al iniciar el backend.

---

## 7. Búsqueda Clásica (A* y Greedy)

Se implementaron dos algoritmos de referencia en `backend/src/engine/search.py` como baseline para evaluar el rendimiento del agente RL.

### 7.1 Greedy (horizonte de un paso)

Evalúa cada acción permitida con una función heurística de un paso:

```
h(s) = (2 − condición) + recursos × 0.5
```

Con bonificaciones contextuales:
- `+5` si la acción es `reaccionar` y `condición > 0`
- `+3` si la acción es `asignar` y `recursos < 5`
- `+2` si la acción es `mover` y `condición == 0`

Elige la acción con mayor `h` sin ningún tipo de exploración o memoria entre pasos.

### 7.2 A* (expansión best-first, profundidad 3)

Expansión limitada a profundidad 3 con máximo 50 nodos visitados. Evalúa caminos combinando la heurística acumulada con ruido `Uniform(0, 0.5)` para romper empates. Retorna la primera acción del mejor camino encontrado. No aprende entre episodios.

### 7.3 Evaluación comparativa

Ambos algoritmos se evalúan sobre entornos independientes (instancias frescas de `Environment`), sin afectar la sesión RL activa. Se ejecutan 10 episodios por algoritmo. El endpoint `GET /api/simulation/comparison` expone:

- **RL:** promedio de reward de los últimos 10 episodios reales del agente entrenado
- **Greedy:** promedio de 10 episodios en entorno fresco
- **A*:** promedio de 10 episodios en entorno fresco

---

## 8. Sistema de Persistencia y Sesiones Continuas

### 8.1 Motivación

Las sesiones largas de entrenamiento pueden superar la memoria disponible del navegador y requerir pausas. El sistema implementa persistencia completa para que el entrenamiento pueda interrumpirse y reanudarse sin pérdida de progreso.

### 8.2 Archivos persistidos

Todos los archivos se almacenan en `backend/data/` (directorio auto-creado, excluido de git):

| Archivo | Contenido | Formato |
|---|---|---|
| `qtable.npy` | Q-tabla 165×4 | NumPy binary |
| `qtable_meta.npy` | Epsilon actual del Q-Learning | NumPy binary |
| `dqn_checkpoint.pt` | Pesos online/target, optimizer, ε, step_count | PyTorch `torch.save` |
| `ml_model.joblib` | RandomForest + LabelEncoder entrenados | joblib |
| `training_log.csv` | Por paso: episode, position, resources, condition, action, reward | CSV append |
| `episode_rewards.csv` | Por episodio: episode_num, total_reward, steps, epsilon, agent_type | CSV append |

Los archivos CSV se escriben en modo **append** (nunca se sobreescriben), acumulando datos de todas las sesiones históricas.

### 8.3 Ciclo de vida de la persistencia

**Al arrancar el backend (lifespan de FastAPI):**

```python
_agent_ql.load(DATA_DIR / "qtable.npy")          # restaura Q-tabla y ε
_agent_dqn.load(DATA_DIR / "dqn_checkpoint.pt")  # restaura red neuronal completa
_predictor.load(DATA_DIR / "ml_model.joblib")     # restaura predictor ML
historical = _analytics.load_csv(DATA_DIR)         # carga historial de episodios
if historical:
    _analytics._episode_rewards = historical
    _analytics.episode = historical[-1]["episode_num"] + 1  # continúa numeración
```

**Auto-guardado configurable durante la simulación:**

```python
# Al completar cada episodio en el handler WebSocket:
if _analytics.episode - _last_saved_episode >= _AUTO_SAVE_EVERY:
    _do_save()
    _last_saved_episode = _analytics.episode
```

El intervalo `_AUTO_SAVE_EVERY` es configurable desde el dashboard (1 a 20 episodios).

**Al apagar el backend:** `_do_save()` en el hook de shutdown del lifespan, garantizando que ningún dato se pierda.

### 8.4 Control de memoria en sesiones largas

El módulo Analytics implementa un cap de memoria para prevenir el agotamiento de RAM en sesiones de miles de pasos:

```python
_RECORDS_MEMORY_CAP = 1_000  # máximo de registros de pasos en memoria

# Tras cada flush a CSV, los registros más antiguos se descartan:
if len(self._records) > _RECORDS_MEMORY_CAP:
    discard = len(self._records) - _RECORDS_MEMORY_CAP
    self._records = self._records[discard:]
    # Los índices de referencia se ajustan para mantener consistencia
```

Los datos descartados de memoria ya están en `training_log.csv` en disco. Las estadísticas (`summary()`) siguen siendo correctas porque se calculan con acumuladores independientes de `_records`.

### 8.5 Modal de inicio: continuar vs. reiniciar

Al iniciar la aplicación, si existen datos persistidos, se presenta un modal que muestra el estado previo (episodios, pasos, epsilon, agente) y ofrece dos opciones:

- **Continuar entrenamiento** (recomendado): carga todos los datos y continúa desde donde se dejó.
- **Iniciar desde cero**: llama a `POST /simulation/reset-full`, que borra todos los archivos persistidos y reinicia los agentes.

### 8.6 Comportamiento por tipo de operación

| Operación | Q-tabla | DQN | ML | CSV histórico |
|---|---|---|---|---|
| `POST /simulation/reset` | ✓ limpia | ✓ limpia | Conservado | **Preservado** |
| `POST /simulation/agent-type` | — (conserva) | — (conserva) | — | **Preservado** |
| `POST /simulation/reset-full` | ✓ limpia | ✓ limpia | ✓ limpia | **Eliminado** |

---

## 9. Optimizaciones de Rendimiento

### 9.1 Problema y contexto

La simulación corre a ~6.7 pasos/segundo (un step cada 150 ms). Cada paso ejecuta: lógica simbólica + ML + agente RL + aprendizaje + analytics + serialización WebSocket. El objetivo es que la pipeline completa tome < 1 ms (headroom ≥ 150×).

### 9.2 Tabla de optimizaciones implementadas

| Componente | Problema original | Solución implementada | Mejora medida |
|---|---|---|---|
| `filter_actions` | Evaluación de predicados Prolog/Python en cada paso | LUT precomputada para los 165 estados | **28× más rápido** |
| `get_active_rules` | Misma evaluación repetida | LUT precomputada | **28× más rápido** |
| `Analytics.summary()` | Reconstrucción de DataFrame Pandas O(n) cada 3 s | Acumuladores incrementales O(1) | **~4 000× más rápido a 10 000 pasos** |
| `Analytics._records` | Crecimiento ilimitado → OOM del proceso | Cap en 1 000 entradas post-flush CSV | **Memoria acotada** |
| WS payload | `q_values` + `rules_info` en cada mensaje (~60% del tamaño) | Envío solo cada 5 pasos; frontend fusiona con último valor | **~60% reducción de payload** |
| `setHistory` en React | `setState` en cada mensaje WS → 6.7 re-renders/s | Batch en ref; flush al estado cada 500 ms | **Re-renders: 6.7/s → 2/s** |
| Bundle JS | Un chunk de 651 KB con Recharts | `manualChunks: { "vendor-charts": ["recharts"] }` en Vite | **244 KB + 405 KB separados (sin warning)** |
| `EpisodeChart` / `RewardChart` | Re-render al cambiar `step` (cada 150 ms) | `React.memo` — solo re-renderiza al cambiar `stats` | **~25× menos re-renders** |
| Selects controlados | Flickering visible mientras API responde | Estado local optimista + sync desde server via `useEffect` | **Sin flickering** |

### 9.3 Benchmarks automatizados

El proyecto incluye `backend/tests/test_performance.py` con **31 pruebas en 5 clases**:

| Clase de test | Qué mide | Umbral mínimo |
|---|---|---|
| `TestFilterActions` | Speedup LUT vs evaluación directa; correctness | ≥ 3× speedup |
| `TestGetActiveRules` | Speedup LUT; completitud de resultados | ≥ 3× speedup |
| `TestAnalyticsSummary` | Latencia de `summary()` en sesión larga | ≤ 1 ms |
| `TestRLAgentThroughput` | Pasos Q-Learning por segundo | ≥ 100 000 pasos/s |
| `TestFullPipeline` | Headroom vs cadencia WebSocket (150 ms) | ≥ 50× headroom |

**Resultados reales en hardware de desarrollo (Windows, CPython 3.12):**

- Q-Learning: ~159 000 pasos/s → headroom **~14 900×** vs cadencia de 6.7 pasos/s
- DQN: ~800 pasos/s → headroom **~120×** (limitado por torch, sigue superando el umbral)
- `summary()`: 0.002 ms constante independiente del número de pasos
- LUT speedup real: **28×** en el hot path

---

## 10. Dashboard de Visualización

### 10.1 Layout de 4 zonas

El frontend usa un grid CSS de 4 zonas fijas:

```
┌─────────────────────────────────────────────────┐
│  TOP HEADER  (42px, full width)                  │
│  Logo · badges WS/Prolog/ML · EP · ε            │
├──────────┬───────────────────────────┬───────────┤
│  LEFT    │       MAIN CONTENT        │   RIGHT   │
│ SIDEBAR  │  (cambia con la nav)      │   PANEL   │
│  200px   │                           │   280px   │
└──────────┴───────────────────────────┴───────────┘
```

El contenido principal cambia completamente según la vista seleccionada en la navegación, sin recargar la página.

### 10.2 Panel izquierdo (LeftSidebar)

| Sección | Descripción |
|---|---|
| **Navegación** | 5 vistas: Operaciones · Entrenamiento · Reglas Prolog · Comparación · Histórico |
| **Red — 5 sectores** | Estado visual de S0-S4 con barra de capacidad MW y resaltado del sector activo |
| **Dinámica activa** | Condición actual del entorno con animación de parpadeo en estados de alerta/fallo |
| **Estadísticas** | Pasos totales · Episodios · Media últimos 10 episodios · Capacidad MW promedio |
| **Sistema** | Política activa · Episodio actual · Estado Prolog · Estado ML · Estado WebSocket |

### 10.3 Panel derecho (RightPanel)

| Sección | Descripción |
|---|---|
| **Métricas clave** | 4 tarjetas: Tasa éxito % · Reward avg · Violaciones Prolog · Epsilon ε |
| **Exploración ε** | Barra de progreso visual del epsilon actual |
| **Entrenamiento en vivo** | 4 sparklines con área rellena, actualizados cada 500 ms |
| **DQN Telemetría** | Loss MSE y tamaño del buffer (solo cuando agente = DQN) |
| **Agente RL** | Botones de selección Q-Learning / DQN |
| **Auto-guardado** | Último episodio guardado · Alerta de episodios sin guardar · Selector de intervalo · Botón "Pausar y guardar" |
| **Control** | Botones ▶ Iniciar · ⏸ Pausar · ↺ Reset |

**SparkCards — métricas en tiempo real** (ventana de últimos 60 pasos):

| SparkCard | Fuente | Por qué esta métrica |
|---|---|---|
| Reward / paso | `step.reward` (+10/−5) | Variación inmediata del rendimiento |
| Tasa éxito % | Rolling 10 pasos: `reward ≥ 10` | Porcentaje de decisiones correctas recientes |
| Actividad simbólica | Rolling 10 pasos: acciones bloqueadas | Frecuencia de estados conflictivos |
| Capacidad MW | `state.resources` (0-10) | Salud de los recursos en tiempo real |

### 10.4 Curva de aprendizaje multi-agente (EpisodeChart)

La vista "Histórico" muestra una curva de aprendizaje avanzada que combina:

- **Datos históricos** de `episode_rewards.csv` (sesiones anteriores)
- **Datos de la sesión actual** en tiempo real
- **Línea cyan** para episodios entrenados con Q-Learning
- **Línea púrpura** para episodios entrenados con DQN
- **Media móvil de 5 episodios** (naranja) para ver tendencia sin ruido
- **Marcadores verticales** en cada cambio de agente con etiqueta "Q-Learning → DQN"
- **Contador y mejor episodio** en la esquina inferior derecha

Esto permite analizar el impacto de cada agente y comparar su evolución en una misma sesión continua.

### 10.5 Vistas del contenido principal

| Vista | Componentes |
|---|---|
| **Operaciones** | EnvironmentMap + DecisionAnatomy (reglas activas, Q-valores, modo exploración/explotación) + DecisionLog (consola de pasos) |
| **Entrenamiento** | AgentView (Q-valores actuales en barras) + PolicyView (política aprendida) + selector de agente con telemetría DQN |
| **Reglas Prolog** | RulesPanel con las 10 reglas y su estado activo/inactivo en el estado actual |
| **Comparación** | ComparisonView: RL vs A* vs Greedy con rewards promedio de 10 episodios |
| **Histórico** | RewardChart + EpisodeChart multi-agente + estadísticas de sesión |

---

## 11. Análisis de Convergencia

### 11.1 Fases del entrenamiento (Q-Learning)

```
Fase 1 — Exploración (pasos 0 – 200, ε: 1.0 → 0.36):
  El agente explora aleatoriamente. Rewards altamente variables.
  Ocasionalmente aparecen picos por secuencias afortunadas.

Fase 2 — Transición (pasos 200 – 600, ε: 0.36 → 0.05):
  Balance entre explorar y explotar. Rewards en tendencia ascendente.
  La Q-tabla empieza a diferenciarse de cero significativamente.

Fase 3 — Convergencia (pasos 600+, ε = 0.05):
  Explotación dominante. Política estabilizada.
  Reward por episodio converge al rango 900–1 400 según volatilidad del entorno.
```

### 11.2 Q-Learning vs DQN — análisis comparativo

| Aspecto | Q-Learning | DQN |
|---|---|---|
| Representación | Tabla explícita 165×4 | Red neuronal 3→64→64→4 |
| Generalización | No (un valor exacto por estado) | Sí (interpola entre estados similares) |
| Estabilidad inicial | Alta (la tabla es estable) | Baja (buffer vacío → sin gradiente) |
| Requisito mínimo | 1 transición para aprender | 64 transiciones (batch size) |
| Volatilidad | Baja (sawtooth suave) | Alta (posible catastrophic interference) |
| Velocidad de convergencia | Rápida para 165 estados | Más lenta; requiere centenas de episodios |
| Escalabilidad | Lineal con |S| × |A| | Escala a espacios continuos y grandes |
| Mejor reward alcanzable | Similar | Ligeramente superior a largo plazo |

### 11.3 Observaciones sobre la curva multi-agente

La curva de aprendizaje registrada en sesiones reales muestra un patrón consistente al cambiar de agente:

- **Transición Q-Learning → DQN:** caída inmediata de reward (2–10 episodios) mientras el buffer se llena. La red inicializada aleatoriamente produce acciones subóptimas.
- **DQN en régimen estable:** mayor varianza episodio a episodio que Q-Learning (el sampling aleatorio del buffer introduce ruido). Los mejores episodios de DQN superan los de Q-Learning.
- **Retorno a Q-Learning:** la Q-tabla conservada del entrenamiento anterior permite recuperar el rendimiento previo rápidamente (1–3 episodios).

Este comportamiento es consistente con la literatura: DQN es superior en espacios de estado grandes o continuos; en espacios discretos pequeños como este (165 estados), Q-Learning es más eficiente.

---

## 12. Resultados Comparativos

### 12.1 Resultados observados (sesión de 181 episodios)

Los siguientes resultados provienen de una sesión real registrada en el dashboard con dos cambios de agente (Q-Learning → DQN en ep. 67, DQN → Q-Learning en ep. 160):

| Período | Agente | Reward promedio | Reward mejor episodio | Observaciones |
|---|---|---|---|---|
| Ep. 1–67 | Q-Learning | 900–1 350 | ~1 350 | Convergencia estable y rápida |
| Ep. 67–160 | DQN | 600–1 500 | ~1 500 | Mayor varianza; valle en ep. 100–106 |
| Ep. 160–181 | Q-Learning | 900–1 640 | **1 640** | Recuperación inmediata; nuevo máximo |
| **Total** | **Combinado** | — | **1 640** | Mejor resultado de la sesión completa |

El valle pronunciado en ep. 100–106 durante la fase DQN corresponde al fenómeno de *catastrophic interference*: la red sobreajusta temporalmente a experiencias recientes. La media móvil confirma que fue transitorio.

### 12.2 Comparación RL vs búsqueda clásica

Evaluación sobre 10 episodios frescos (entornos independientes), agente RL en fase de explotación (ε ≈ 0.05):

| Algoritmo | Tipo | Reward promedio estimado | Aprende con el tiempo |
|---|---|---|---|
| Q-Learning (entrenado, 100+ ep.) | RL tabular | ~900–1 350 | ✓ Sí |
| DQN (entrenado, 100+ ep.) | RL neuronal | ~800–1 500 | ✓ Sí |
| A* (profundidad 3) | Búsqueda clásica | ~250–380 | ✗ No |
| Greedy (horizonte 1) | Búsqueda clásica | ~200–350 | ✗ No |

**Ventaja del RL sobre búsqueda clásica:**

El agente RL aprende las consecuencias de largo plazo de sus acciones. En estados donde la acción inmediatamente rentable (`asignar_recursos`) no es óptima a largo plazo (porque agota el turno y la condición sigue deteriorándose), el RL aprende a preferir `reaccionar` primero — algo que ni Greedy ni A* con horizonte corto pueden descubrir.

### 12.3 Efectividad de la lógica simbólica

Sin las reglas Prolog, el agente puede desperdiciar acciones en estados donde son claramente contraproducentes (e.g., `mover` en condición crítica, `reaccionar` en red estable). Con las 10 reglas activas:

- El espacio efectivo de acciones se reduce en estados conflictivos, acelerando el aprendizaje.
- Los rewards negativos evitables (−5 por acción incorrecta) disminuyen en las fases tempranas.
- El agente converge más rápido porque las acciones aleatorias en exploración ya son sub-óptimas conocidas que Prolog descarta.
- La "Actividad simbólica" en el dashboard cuantifica cuántos pasos tienen al menos una acción bloqueada — un indicador de cuán desafiante es el estado actual.

### 12.4 Papel del predictor ML

La distribución de probabilidad que el predictor ML añade al panel de "Decisión Anatómica" no interviene directamente en la selección de acción del agente RL en la versión actual — sirve como información contextual para el operador. Sin embargo, su uso futuro como feature adicional de estado podría mejorar el rendimiento en entornos más complejos.

---

## 13. Stack Tecnológico

### Backend

| Herramienta | Versión | Rol |
|---|---|---|
| Python | 3.12 | Lenguaje principal |
| uv | 0.11+ | Gestión de dependencias y entorno virtual (10-100× más rápido que pip) |
| FastAPI | 0.115+ | API REST + WebSocket con soporte async nativo |
| PyTorch | 2.5+ | Red neuronal DQN (CPU; escalable a GPU) |
| scikit-learn | 1.5+ | RandomForestClassifier para predicción de entorno |
| Pandas / NumPy | 2.x | Analytics, persistencia CSV, álgebra lineal |
| SWI-Prolog | 10.0.2 | Motor de lógica simbólica (ISO Prolog) |
| pyswip | 0.3.3 | Binding Python ↔ SWI-Prolog |
| joblib | 1.4+ | Serialización de modelos scikit-learn |
| Ruff | 0.7+ | Linting + formato (binario Rust, reemplaza flake8+black+isort) |
| pytest | 8.3+ | 31 tests de rendimiento + tests unitarios |

### Frontend

| Herramienta | Versión | Rol |
|---|---|---|
| React | 19 | UI reactiva con hooks |
| TypeScript | 5.x | Tipado estático end-to-end |
| Vite | 6 | Build + servidor de desarrollo con HMR |
| Bun | 1.3+ | Gestor de paquetes (sin scripts de dependencias por defecto) |
| Recharts | 2.x | RewardChart y EpisodeChart (bundle separado via manualChunks) |
| Biome | 1.9 | Lint + formato (binario Rust, reemplaza ESLint+Prettier) |

---

## 14. Decisiones de Diseño

**Seguridad de dependencias (Bun):** Bun no ejecuta scripts `preinstall`/`postinstall` de paquetes de terceros por defecto, mitigando ataques de supply chain. Se configuró `minimumReleaseAge = 259200` (3 días) en `bunfig.toml` para bloquear paquetes recién publicados.

**uv sobre pip:** lockfile auditable (`uv.lock`), entorno virtual automático y aislado, sin conflictos de versiones globales.

**Ruff y Biome sobre herramientas tradicionales:** cada uno reemplaza múltiples herramientas con un único binario Rust, eliminando dependencias transitivas potencialmente vulnerables.

**LUT vs evaluación en tiempo de ejecución:** dado que el espacio de estados es finito y pequeño, precomputar todos los resultados posibles al iniciar el módulo es la estrategia óptima. La alternativa (evaluar predicados Prolog en cada paso) sería funcionalmente equivalente pero 28× más lenta.

**Acumuladores vs DataFrame Pandas:** `summary()` se llama cada 3 s mientras la simulación corre. Reconstruir un DataFrame completo en cada llamada escala O(n) con el número de pasos. Los acumuladores incrementales mantienen la operación en O(1) independientemente de la duración de la sesión.

**Batching del historial en el frontend:** React re-renderiza componentes cuando su estado cambia. Acumular los pasos WebSocket en un `ref` y flushar al estado React cada 500 ms reduce los re-renders de gráficos de 6.7/s a 2/s, sin afectar la visualización del estado actual del agente (que se actualiza inmediatamente).

**Persistencia CSV append-only:** los archivos CSV nunca se sobreescriben — solo se agregan filas nuevas. Esto garantiza que un crash o interrupción no corrompe el historial y que varias sesiones se acumulan naturalmente en los mismos archivos para el análisis histórico.

---

## 15. Conclusiones

El sistema implementado demuestra que la integración de múltiples paradigmas de IA produce resultados superiores a cualquier enfoque aislado:

1. **El aprendizaje por refuerzo** (Q-Learning y DQN) supera a la búsqueda clásica en sesiones de entrenamiento de 50+ episodios, porque aprende las consecuencias a largo plazo de las acciones en función del historial acumulado de interacciones.

2. **La lógica simbólica en Prolog** complementa al RL de manera fundamental: las 10 reglas en `rules.pl` (cláusulas Horn con negación por fallo) eliminan acciones claramente incorrectas del espacio de decisión en cada paso, acelerando la convergencia y reduciendo los rewards negativos evitables. La integración Python ↔ SWI-Prolog via `pyswip` y el fallback automático a predicados Python garantizan disponibilidad en cualquier entorno.

3. **El modelo ML predictivo** (RandomForest) añade anticipación al sistema con reentrenamiento online cada 50 pasos, sin interrumpir la simulación.

4. **La búsqueda clásica** (A*, Greedy) establece un baseline interpretable y cuantificable: el agente RL supera ambos algoritmos consistentemente una vez que ε < 0.3, validando que el aprendizaje es genuino y no producto de exploración aleatoria.

5. **Las optimizaciones de rendimiento** (LUT O(1), acumuladores, batching, code splitting) hacen que el sistema sea viable para sesiones de entrenamiento prolongadas: el headroom entre la velocidad de la pipeline y la cadencia WebSocket es ~14 900×, con memoria del proceso acotada independientemente de la duración de la sesión.

6. **El sistema de persistencia** permite continuar el entrenamiento entre sesiones, reanudar exactamente en el mismo estado de epsilon y pesos del agente, y visualizar la curva de aprendizaje histórica acumulada con diferenciación por tipo de agente.

**Q-Learning es más eficiente** para este dominio concreto de 165 estados por su convergencia rápida y estabilidad. **DQN ofrece mayor escalabilidad** para versiones futuras del problema con espacios de estado más grandes o variables continuas, y puede alcanzar rewards máximos superiores tras sesiones de entrenamiento más largas.

---

*Tests:* 31/31 pasando · *Build:* backend uv + frontend Bun · *Prolog:* SWI-Prolog 10 + pyswip · *Sesión de referencia:* 181 episodios · mejor reward: 1 640
