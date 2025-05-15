# src/utils.py

import numpy as np


def run_and_render_episode(env, policy_fn, max_steps=1000):
    """
    Ejecuta un episodio con la política dada y devuelve la lista de frames renderizados.
    policy_fn(obs) -> action
    """
    frames = []
    obs, _ = env.reset()
    done = False
    steps = 0
    total_reward = 0

    while not done and steps < max_steps:
        # Captura frame actual
        frame = env.render()
        frames.append(frame)

        # Elige acción (en función de la política dada)
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward  # Acumula la recompensa
        done = terminated or truncated
        steps += 1

    # Captura frame final
    if not done:
        frames.append(env.render())

    return frames, total_reward


def random_policy(obs):
    # Política aleatoria: hit (1) con prob 0.5, stick (0) con prob 0.5
    return np.random.choice([0, 1])
