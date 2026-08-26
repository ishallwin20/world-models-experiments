# data.py

import numpy as np
import torch

from env import MovingSquareEnv, ACTION_DIM


def generate_transitions(
    num_episodes: int = 64,
    steps_per_episode: int = 100,
    size: int = 32,
    square_size: int = 4,
    speed: float = 2.0,
    seed: int = 0,
):
    env = MovingSquareEnv(
        size=size,
        square_size=square_size,
        speed=speed,
        max_steps=steps_per_episode,
    )

    frames = []
    actions = []
    next_frames = []

    for episode in range(num_episodes):
        obs = env.reset(seed=seed + episode)

        for step in range(steps_per_episode):
            action = int(env.rng.integers(0, ACTION_DIM))

            next_obs, reward, done, info = env.step(action)
            
            frames.append(obs.copy())
            actions.append(action)
            next_frames.append(next_obs.copy())

            obs = next_obs

            if done:
                obs = env.reset()

    X = torch.from_numpy(np.stack(frames, axis=0, dtype=np.float32))
    A = torch.from_numpy(np.stack(actions, axis=0, dtype=np.int64))
    Y = torch.from_numpy(np.stack(next_frames, axis=0, dtype=np.float32))

    return X, A, Y