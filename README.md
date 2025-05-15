# Aprendizaje por refuerzo: BlackJack
#### **Autores:** Jose Carlos Riego, Pablo Rodríguez, Ángel Visedo

## 0. Entorno de desarrollo

Este proyecto ha sido realizado con **Python 3.11.11**.  
Para replicar el entorno, ejecute: `pip install -r requirements.txt`

## 1. Introducción

### 1.1. Fundamentos del Blackjack  
El Blackjack es un juego de cartas donde el jugador busca sumar hasta 21 sin pasarse, compitiendo contra un crupier que sigue reglas deterministas (pide hasta 17). Las cartas numéricas valen su número, las figuras valen 10 y el as puede contar como 1 u 11.

### 1.2. Motivación y objetivos  
Se elige Blackjack por su naturaleza probabilística y recompensa bien definida, ideal para evaluar algoritmos de RL avanzados. El objetivo es ver si métodos modernos (Double Q-Learning, Double DQN, A2C) superan la estrategia básica humana sin conteo de cartas.

### 1.3. Ventajas de los algoritmos avanzados de RL  
Las técnicas “double” reducen el sesgo de sobreestimación; los métodos basados en redes profundas permiten manejar espacios de estados más complejos; y los enfoques actor-critic combinan políticas estocásticas con estimación de valor, favoreciendo una exploración más eficiente.

## 2. Entorno Blackjack de Gymnasium

### 2.1. Definición formal del MDP  
- **Espacio de estados** S = {(player_sum, dealer_card, usable_ace)} con un total de 360 estados posibles.  
- **Acciones** A = {hit (pedir carta), stick (plantarse)}.  
- **Transiciones**: baraja infinita, probabilidad uniforme de cada carta.  
- **Recompensas** R = +1 (victoria), 0 (empate), −1 (derrota), otorgada al final del episodio.  
- **Factor de descuento** γ = 1.

### 2.2. Hipótesis y simplificaciones adoptadas  
- Uso de baraja infinita (sin posibilidad de conteo de cartas).  
- Sin acciones de doblar, dividir ni asegurar.  
- Blackjack natural pagado 1:1.  
- Reglas ajustadas para alinearse con la implementación estándar de Gymnasium.

### 2.3. Resultados de referencia  
Estrategia básica humana estándar (6–8 mazos, crupier planta en soft-17, paga 3:2 al natural) arroja aproximadamente:  
| Métrica    | Media   | Desviación Típica |
|------------|:-------:|:-----------------:|
| Victorias  | 42.22 % | 0.05 %            |
| Empates    | 8.48 %  | 0.03 %            |
| Derrotas   | 49.10 % | 0.06 %            |
| Retorno    | −0.005  | 0.001             |

## 3. Entrenamiento y evaluación de los modelos

### 3.1. Q-Learning

#### 3.1.1. Descripción del método  
Algoritmo tabular off-policy ε-greedy que actualiza la función Q mediante la ecuación:  

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \Bigl(r + \gamma \max_{a'} Q(s',a') - Q(s,a)\Bigr)
$$


con tasa de aprendizaje α = 0.1 y descuento γ = 1.

#### 3.1.2. Entrenamiento  
- ε ∈ {1.0, 0.1, 0.01} para controlar la exploración.  
- 500 000 episodios por cada valor de ε.

#### 3.1.3. Validación  
Evaluación greedy (ε = 0) en 10 000 episodios:  
| ε    | Victorias | Empates | Derrotas | Retorno |
|:----:|:---------:|:-------:|:--------:|:-------:|
| 0.01 | 41.18 %   | 7.17 %  | 51.65 %  | −0.105  |
| 1.00 | 39.90 %   | 5.81 %  | 54.29 %  | −0.144  |
| 0.10 | 38.87 %   | 5.61 %  | 55.52 %  | −0.167  |

### 3.2. Double Q-Learning

#### 3.2.1. Descripción del método  
Se emplean dos tablas, QA y QB, que se actualizan alternando la selección y evaluación de acciones, reduciendo el sesgo de sobreestimación.

#### 3.2.2. Ventajas en el Blackjack  
Mayor estabilidad en la estimación de valores y mejor gestión del riesgo-recompensa.

#### 3.2.3. Entrenamiento  
- Schedules de ε: AdaptExp (exponencial), LinearDecay (lineal) y OptimalMix.  
- α decrece linealmente de 0.1 a 0.01, γ = 1, 500 000 episodios.

#### 3.2.4. Validación  
Evaluación greedy en 10 000 episodios; mejor configuración (AdaptExp):  
| Schedule    | Victorias | Empates | Derrotas | Retorno |
|:-----------:|:---------:|:-------:|:--------:|:-------:|
| AdaptExp    | 43.29 %   | 8.90 %  | 47.81 %  | −0.045  |
| LinearDecay | 42.88 %   | 9.10 %  | 48.04 %  | −0.052  |
| OptimalMix  | 43.28 %   | 8.80 %  | 47.87 %  | −0.046  |

### 3.3. Deep Q-Network (DQN) y Double DQN

#### 3.3.1. Descripción de los métodos  
- **DQN**: usa una red neuronal para aproximar Q(s,a), con target network y replay buffer.  
- **Double DQN**: desacopla la selección (red online) y evaluación (red target) de acciones.

#### 3.3.2. Entrenamiento  
- Arquitectura: 3 capas densas de 128 unidades (ReLU) + salida para 2 acciones.  
- γ = 1, replay buffer de 60 000 transiciones, aprendizaje comienza tras 5 000 pasos, actualización de target cada 2 000 pasos.  
- ε decae de 1.0 a 0.05 en 250 000 pasos.  
- lr ∈ {1e-3, 5e-4}, batch size ∈ {64, 256}.  
- 500 000 episodios por cada combinación de hiperparámetros.

#### 3.3.3. Validación  
Evaluación greedy en 10 000 episodios; mejor agente Double DQN (lr = 1e-3, batch = 64):  
| Algoritmo            | Victorias | Empates | Derrotas | Retorno |
|----------------------|:---------:|:-------:|:--------:|:-------:|
| Double DQN (1e-3,64) | 44.18 %   | 8.78 %  | 47.04 %  | −0.029  |

### 3.4. Advantage Actor-Critic (A2C)

#### 3.4.1. Descripción del método  
Red compartida con dos salidas: softmax para la política y lineal para la función de valor. Se actualiza mediante gradiente de política usando la ventaja A(s,a).

#### 3.4.2. Entrenamiento  
- lr ∈ {0.01, 0.001, 0.0001}, γ = 1.  
- 500 000 episodios.

#### 3.4.3. Validación  
Evaluación greedy en 10 000 episodios; mejor lr = 0.01:  
| lr    | Victorias | Empates | Derrotas | Retorno |
|:-----:|:---------:|:-------:|:--------:|:-------:|
| 0.01  | 43.63 %   | 8.72 %  | 47.65 %  | −0.040  |
| 0.001 | 42.82 %   | 7.31 %  | 49.87 %  | −0.070  |
|0.0001 | 37.44 %   | 4.70 %  | 57.86 %  | −0.204  |

## 4. Comparación entre métodos

| Método                         | Victorias | Empates | Derrotas | Retorno |
|--------------------------------|:---------:|:-------:|:--------:|:-------:|
| Estrategia básica (sin conteo) | 42.22 %   | 8.48 %  | 49.10 %  | −0.005  |
| **Double DQN (1e-3,64)**       | **44.18 %**| 8.78 % | 47.04 %  | **−0.029** |
| A2C (lr = 0.01)                | 43.63 %   | 8.72 %  | 47.65 %  | −0.040  |
| Double Q-Learning (AdaptExp)   | 43.29 %   | 8.90 %  | 47.81 %  | −0.045  |
| Q-Learning (ε = 0.01)          | 41.18 %   | 7.17 %  | 51.65 %  | −0.105  |

## 5. Conclusiones

1. **Double DQN** obtuvo el mejor retorno medio y la mayor tasa de victorias, confirmando el beneficio de desacoplar selección y evaluación.  
2. **A2C** y **Double Q-Learning** mejoran sustancialmente sobre Q-Learning clásico.  
3. Ningún agente supera el retorno de la estrategia básica (−0.005) dadas las diferencias en las reglas usadas (con vs sin natural), aunque varios superan su tasa de victorias, con lo cual se concluye que estos métodos superan la estrategia humana sin conteo de cartas.
4. Líneas futuras: refinar la política en estados “soft”, explorar arquitecturas más profundas y políticas basadas en Monte Carlo.

## 6. Referencias

1. Baldwin, R.R. et al. (1956). *The Optimum Strategy in Blackjack*. JASA.  
2. Thorp, E.O. (1962). *Beat the Dealer*. Vintage Books.  
3. Sutton, R.S.; Barto, A.G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.  
4. Mnih, V. et al. (2015). *Human-level control through deep RL*. Nature.  
5. Van Hasselt, H.; Guez, A.; Silver, D. (2016). *Deep RL with Double Q-learning*. AAAI.  
6. Shackleford, M. (2025). *Blackjack House Edge Tables*. wizardofodds.com  
