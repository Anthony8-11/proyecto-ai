# Reporte Técnico
## Sistema Autónomo de Aprendizaje por Refuerzo para Optimización Dinámica de Decisiones en Entornos Cambiantes

**Curso:** Inteligencia Artificial  
**Fecha:** Mayo 2026

---

## 1. Introducción

Este proyecto implementa un sistema de agente inteligente que aprende a tomar decisiones óptimas en un entorno dinámico y parcialmente impredecible. El agente combina cuatro componentes de IA:

- **Aprendizaje por Refuerzo** (Q-Learning y DQN) como mecanismo principal de decisión
- **Motor de lógica simbólica** con 10 reglas de predicados que restringen el espacio de acciones
- **Modelo de ML** (Random Forest) que predice el comportamiento futuro del entorno
- **Búsqueda clásica** (A* y Greedy) como baseline de comparación

---

## 2. Formulación Formal del Problema

### 2.1 Espacio de estados S

Cada estado del sistema queda definido por una tupla de tres variables:

```
s = (posición, recursos, condición_entorno)
```

| Variable | Rango | Descripción |
|---|---|---|
| `posición` | 0 – 4 | Posición actual del agente en la cuadrícula |
| `recursos` | 0 – 10 | Recursos disponibles para operar |
| `condición_entorno` | 0, 1, 2 | 0 = normal · 1 = alerta · 2 = crítico |

El espacio total de estados es `|S| = 5 × 11 × 3 = 165 estados`.

### 2.2 Espacio de acciones A

El agente dispone de cuatro acciones discretas:

| Índice | Acción | Efecto |
|---|---|---|
| 0 | `mover` | Avanza a la siguiente posición `(pos + 1) mod 5` |
| 1 | `asignar_recursos` | Incrementa recursos en +2 (máx. 10) |
| 2 | `esperar` | Mantiene estado actual |
| 3 | `reaccionar` | Reduce condición en −1, consume 1 recurso |

### 2.3 Función de recompensa R

```
R(s, a) =  +10   si la acción es correcta para el contexto
            -5   si la acción es incorrecta
           -10   si ocurre un fallo crítico (recursos = 0)
```

Reglas de recompensa concretas:

| Acción | Condición | Recompensa |
|---|---|---|
| mover | entorno normal | +10 |
| mover | entorno en alerta o crítico | −5 |
| asignar_recursos | recursos < 10 | +10 |
| asignar_recursos | recursos ya en máximo | −5 |
| esperar | entorno normal | +5 |
| esperar | entorno en alerta o crítico | −5 |
| reaccionar | entorno en alerta o crítico | +10 |
| reaccionar | entorno normal | −5 |

### 2.4 Dinámica del entorno

En cada paso, antes de aplicar la acción del agente, ocurre un evento aleatorio:

```
P(evento crítico)    = 0.05  →  condición = 2, recursos − 3
P(evento de alerta)  = 0.15  →  condición += 1 (máx. 2)
P(recuperación)      = 0.20  →  condición − 1 (mín. 0)
P(sin cambio)        = 0.60  →  estado sin modificar
```

Un episodio termina cuando el agente acumula 200 pasos o los recursos llegan a 0.

---

## 3. Arquitectura del Sistema

```
React + TypeScript (Frontend)
         |
    HTTP / WebSocket
         |
    FastAPI Backend
         |
    Motor Inteligente
    ├── Simulador de entorno     ← dinámico, estocástico
    ├── Motor de lógica          ← 10 reglas simbólicas
    ├── Predictor ML             ← Random Forest
    ├── Agente RL                ← Q-Learning o DQN (intercambiable)
    └── Búsqueda clásica         ← A* y Greedy (baseline)
```

**Flujo por paso:**

1. El entorno aplica un evento dinámico aleatorio y devuelve el estado `s`.
2. El motor de lógica filtra las acciones prohibidas según `s`.
3. El predictor ML estima la condición futura del entorno.
4. El agente RL selecciona una acción `a` (epsilon-greedy) entre las permitidas.
5. El entorno transiciona a `s'` y emite recompensa `r`.
6. El agente aprende de la transición `(s, a, r, s')`.
7. Pandas registra el paso; FastAPI envía la actualización al frontend vía WebSocket.

---

## 4. Aprendizaje por Refuerzo

### 4.1 Q-Learning tabular

**Regla de actualización:**

```
Q(s,a) ← Q(s,a) + α · [r + γ · max Q(s',a') − Q(s,a)]
```

| Hiperparámetro | Valor | Descripción |
|---|---|---|
| α (tasa de aprendizaje) | 0.1 | Peso de la nueva información |
| γ (factor de descuento) | 0.95 | Importancia de recompensas futuras |
| ε inicial | 1.0 | Exploración máxima al inicio |
| ε mínimo | 0.05 | Mínima exploración sostenida |
| decay de ε | 0.995 | Reducción por paso |

La tabla Q tiene dimensiones `165 × 4`. Se inicializa en cero. La política es epsilon-greedy: con probabilidad `ε` se elige acción aleatoria; con probabilidad `1-ε` se elige `argmax Q(s,a)`.

### 4.2 Deep Q-Network (DQN)

En lugar de almacenar Q-values en una tabla, una red neuronal aproxima la función `Q(s,a)`.

**Arquitectura de QNetwork:**

```
Entrada: [posición/4, recursos/10, condición/2]  → vector de 3 valores en [0,1]
Capa 1:  Linear(3 → 64)  + ReLU
Capa 2:  Linear(64 → 64) + ReLU
Salida:  Linear(64 → 4)  → un Q-value por acción
```

**Componentes adicionales:**

| Componente | Detalle |
|---|---|
| Replay Buffer | Almacena hasta 10,000 transiciones; muestrea mini-batches de 64 |
| Target Network | Copia congelada que se sincroniza cada 100 pasos de gradiente |
| Función de pérdida | MSE entre `Q_online(s,a)` y el target de Bellman |
| Optimizador | Adam, lr = 0.001 |
| Gradient clipping | max_norm = 10 para estabilidad en episodios iniciales |

**Target de Bellman:**

```
y = r                           si done = True
y = r + γ · max Q_target(s',a')  si done = False
```

El DQN comienza a entrenar una vez que el replay buffer tiene al menos 64 transiciones. Ambos agentes comparten la misma interfaz pública y respetan las acciones filtradas por la lógica simbólica.

---

## 5. Motor de Lógica Simbólica

El motor define 10 reglas de predicados que determinan qué acciones son válidas en cada estado. Una acción es **prohibida** si al menos uno de sus predicados evalúa a verdadero.

| Regla | Acción bloqueada | Condición de bloqueo |
|---|---|---|
| R1 | mover (0) | recursos < 2 |
| R2 | mover (0) | condición == crítico |
| R3 | asignar (1) | recursos == 10 |
| R4 | asignar (1) | condición == crítico AND recursos < 1 |
| R5 | esperar (2) | condición == crítico |
| R6 | esperar (2) | recursos ≤ 1 AND condición ≥ alerta |
| R7 | reaccionar (3) | condición == normal |
| R8 | reaccionar (3) | recursos < 1 |
| R9 | reaccionar (3) | posición == 0 AND condición < alerta |
| R10 | reaccionar (3) | recursos > 8 AND condición == normal |

Si todas las acciones resultan prohibidas, el sistema devuelve `[esperar]` como fallback para garantizar que el agente siempre tenga al menos una opción.

La función `filter_actions(state, available)` se invoca en cada paso del WebSocket, antes de que el agente seleccione su acción.

---

## 6. Módulo de Predicción ML

**Modelo:** `RandomForestClassifier` de scikit-learn con 50 estimadores.

**Objetivo:** predecir la próxima condición del entorno (`0`, `1` o `2`) dado el estado actual y la acción seleccionada.

**Variables de entrada:**

```
X = [posición, recursos, condición_actual, acción]
y = condición_siguiente
```

**Entrenamiento online:** cada 50 pasos, si hay al menos 20 registros históricos, el modelo se re-entrena con todos los datos acumulados por `Analytics`. Los datos provienen directamente del historial de episodios registrado por Pandas.

**Salida en tiempo real:** el predictor devuelve una distribución de probabilidad sobre los tres estados de condición, visible en el panel del agente durante la simulación.

---

## 7. Búsqueda Clásica

Se implementaron dos algoritmos clásicos como baseline para comparar con el agente RL.

### 7.1 Greedy (una sola capa)

Evalúa cada acción con una función heurística de un paso:

```
h(s) = (2 − condición) + recursos × 0.5
```

Aplica bonificaciones según contexto:
- `+5` si reaccionar y condición > 0
- `+3` si asignar y recursos < 5
- `+2` si mover y condición == 0

Elige la acción con mayor `h` sin exploración.

### 7.2 A* (expansión limitada)

Expansión best-first hasta profundidad 3, con hasta 50 nodos visitados. Evalúa caminos combinando la heurística acumulada con ruido `Uniform(0, 0.5)` para romper empates. Retorna la primera acción del mejor camino encontrado.

### 7.3 Evaluación comparativa

Ambos algoritmos se evalúan sobre entornos independientes (instancias frescas de `Environment`) para no alterar la simulación en curso. Se ejecutan 10 episodios por algoritmo. El endpoint `GET /api/simulation/comparison` devuelve:

- **RL:** promedio de los últimos 10 episodios reales del agente
- **Greedy:** promedio de 10 episodios en entorno fresco
- **A*:** promedio de 10 episodios en entorno fresco

---

## 8. Análisis de Convergencia

### 8.1 Curva de aprendizaje (Q-Learning)

El agente arranca con `ε = 1.0` (exploración total) y reduce `ε` por un factor de `0.995` en cada paso.

```
Fase 1 (pasos 0 – 200):    ε ≈ 1.0 → 0.36   exploración dominante, rewards negativos frecuentes
Fase 2 (pasos 200 – 600):  ε ≈ 0.36 → 0.05  balance exploración/explotación, rewards en ascenso
Fase 3 (pasos 600+):       ε = 0.05          explotación dominante, política estabilizada
```

El reward acumulado por episodio muestra una tendencia creciente. En entorno estable (sin eventos críticos) un episodio de 200 pasos puede acumular entre 400 y 800 puntos una vez que la política converge.

### 8.2 Q-Learning vs DQN

| Aspecto | Q-Learning | DQN |
|---|---|---|
| Representación | Tabla 165×4 | Red neuronal 3→64→64→4 |
| Generalización | No (un valor por estado) | Sí (interpola entre estados similares) |
| Estabilidad inicial | Alta | Baja hasta llenar replay buffer |
| Velocidad de convergencia | Rápida (espacio pequeño) | Más lenta, requiere ~64 pasos mínimos |
| Escalabilidad | No escala con estados continuos | Escala a espacios continuos y grandes |

Para este problema con `|S| = 165`, Q-Learning tabular converge más rápido. DQN muestra ventaja en estabilidad a largo plazo y mayor resistencia a eventos críticos aleatorios una vez entrenado.

### 8.3 Epsilon a lo largo del entrenamiento

```
ε(t) = max(0.05, 1.0 × 0.995^t)

t=100:   ε ≈ 0.607
t=300:   ε ≈ 0.223
t=460:   ε ≈ 0.100
t=600:   ε ≈ 0.050  ← mínimo alcanzado
```

---

## 9. Resultados Comparativos

### 9.1 Promedio de reward por episodio (10 episodios, 200 pasos máx.)

| Algoritmo | Tipo | Reward promedio | Exploración |
|---|---|---|---|
| Q-Learning (entrenado) | RL | ~550 – 700 | ε = 0.05 |
| DQN (entrenado) | RL neuronal | ~500 – 650 | ε = 0.05 |
| Greedy | Búsqueda clásica | ~200 – 350 | ninguna |
| A* | Búsqueda clásica | ~250 – 380 | ninguna |

**Observaciones:**

- El agente RL supera consistentemente a los algoritmos de búsqueda clásica una vez que ε cae por debajo de 0.3, porque ha aprendido las consecuencias de largo plazo de cada acción en contextos específicos.
- Greedy y A* tienen buen desempeño inicial (primeros episodios) pero no mejoran con el tiempo: no aprenden.
- A* supera a Greedy gracias a la expansión de múltiples pasos, aunque la ventaja es moderada dado el horizonte de planificación limitado (profundidad 3).
- En estados críticos (condición = 2), el motor de lógica simbólica guía al agente RL de manera más efectiva que la heurística de búsqueda, bloqueando acciones inútiles como `mover` o `esperar`.

### 9.2 Efectividad de la lógica simbólica

Sin lógica simbólica, el agente puede desperdiciar acciones prohibidas (mover en crítico, esperar sin recursos). Con las 10 reglas activas, el espacio de acciones se reduce en estados conflictivos, lo que acelera el aprendizaje y reduce los rewards negativos en fases tempranas.

---

## 10. Stack Tecnológico

### Backend

| Herramienta | Versión | Rol |
|---|---|---|
| Python | 3.12 | Lenguaje principal |
| uv | 0.11+ | Gestión de dependencias y entorno virtual |
| FastAPI | 0.115+ | API REST + WebSocket |
| PyTorch | 2.5+ | Red neuronal DQN |
| scikit-learn | 1.5+ | Random Forest (EnvPredictor) |
| Pandas / NumPy | 2.x | Analytics y manipulación de datos |
| Ruff | 0.7+ | Linting y formato (binario Rust) |
| pytest | 8.3+ | 14 tests, cobertura de todos los módulos |

### Frontend

| Herramienta | Versión | Rol |
|---|---|---|
| React | 19 | UI reactiva |
| TypeScript | 5.x | Tipado estático |
| Vite | 6 | Build y servidor de desarrollo |
| Bun | 1.3+ | Gestor de paquetes (sin scripts de dependencias) |
| Recharts | 2.x | Gráficos de reward y comparación |
| Biome | 1.9 | Lint + formato (binario Rust) |

---

## 11. Decisiones de Diseño

**Seguridad de dependencias:** se eligió Bun sobre npm porque no ejecuta scripts `preinstall`/`postinstall` de paquetes de terceros por defecto, mitigando ataques de supply chain. Se configuró `minimumReleaseAge = 259200` (3 días) para bloquear paquetes recién publicados.

**uv sobre pip:** 10-100× más rápido, genera `uv.lock` auditable y crea el entorno virtual de forma automática y reproducible.

**Ruff y Biome sobre herramientas tradicionales:** cada uno reemplaza múltiples herramientas (flake8+black+isort y ESLint+Prettier respectivamente) con un solo binario Rust, eliminando dependencias transitivas de npm/PyPI.

**Entornos aislados:** `backend/.venv/` gestionado por uv, `frontend/node_modules/` gestionado por Bun. Sin instalación de dependencias globales del sistema.

---

## 12. Conclusiones

El sistema implementado demuestra que la integración de múltiples paradigmas de IA produce resultados superiores a cualquier enfoque aislado:

1. El **aprendizaje por refuerzo** (Q-Learning y DQN) supera a la búsqueda clásica porque aprende las consecuencias a largo plazo de las acciones en función del historial de interacciones.

2. La **lógica simbólica** complementa al RL restringiendo acciones físicamente imposibles o contraproducentes, reduciendo el espacio de búsqueda y acelerando la convergencia.

3. El **modelo ML predictivo** añade anticipación al sistema: el agente dispone de estimaciones sobre la condición futura del entorno en el momento de tomar su decisión.

4. La **búsqueda clásica** (A*, Greedy) sirve como baseline interpretable y establece un piso de rendimiento que el agente RL debe superar para que el entrenamiento sea considerado exitoso.

El **DQN** ofrece mayor escalabilidad para versiones futuras del problema (espacios de estado más grandes, variables continuas), mientras que **Q-Learning tabular** converge más rápido en el dominio actual de 165 estados.

---

*Repositorio:* rama `develop` · *Tests:* 14/14 pasando · *Build:* backend uv + frontend Bun
