# env.py

import numpy as np

ACTION_DIM = 5

class MovingSquareEnv:
    def __init__(
        self,
        size: int = 32,
        square_size: int = 4,
        speed: float = 2.0,
        max_steps: int = 100,
    ):
        self.size = size
        self.square_size = square_size
        self.speed = speed
        self.max_steps = max_steps
        self.rng = np.random.default_rng()
        self.pos = None
        self.t = 0

    def reset(self, seed=None):
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)
        
        high = self.size - self.square_size
        x = self.rng.integers(0, high + 1)
        y = self.rng.integers(0, high + 1)

        self.pos = np.array([x,y], dtype=np.float32)
        self.t = 0
        return self._render()

    def _action_to_vel(self, action: int) -> np.ndarray:
        if action == 0:
            return np.array([0.0, 0.0], dtype=np.float32)
        if action == 1:
            return np.array([0.0, -self.speed], dtype=np.float32)
        if action == 2:
            return np.array([0.0, self.speed], dtype=np.float32)
        if action == 3:
            return np.array([-self.speed, 0.0], dtype=np.float32)
        if action == 4:
            return np.array([self.speed, 0.0], dtype=np.float32)

        raise ValueError(f"Invalid action: {action}")

    def step(self, action: int):
        vel = self._action_to_vel(action)
        self.pos = np.clip(
            self.pos + vel,
            0.0,
            self.size - self.square_size,
        )

        self.t += 1
        done = self.t >= self.max_steps

        return self._render(), 0.0, done, {}

    def _render(self):
        frame = np.zeros((1, self.size, self.size), dtype=np.float32)

        x = int(np.round(self.pos[0]))
        y = int(np.round(self.pos[1]))

        x = max(0, min(x, self.size - self.square_size))
        y = max(0, min(y, self.size - self.square_size))

        frame[
            0,
            y:y + self.square_size,
            x:x + self.square_size,
        ] = 1.0

        return frame
