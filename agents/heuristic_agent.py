import numpy as np
from itertools import product
import random

class Agent:
    """
    Heuristic Minesweeper agent:
    - Picks tiles adjacent to revealed 0s first (guaranteed safe)
    - Otherwise picks randomly among exposed tiles (-1)
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, obs):
        H, W, D = obs.shape
        frontier = np.argwhere(obs == -1)
        zeros = np.argwhere(obs == 0)

        for zx, zy, zz in zeros:
            for dx, dy, dz in product((-1,0,1), repeat=3):
                if dx == dy == dz == 0:
                    continue
                nx, ny, nz = zx + dx, zy + dy, zz + dz
                if 0 <= nx < H and 0 <= ny < W and 0 <= nz < D:
                    if obs[nx, ny, nz] == -1:
                        print("Picking safe:", (nx, ny, nz))
                        action = nz * (H * W) + ny * W + nx
                        return action

        if len(frontier) > 0:
            x, y, z = random.choice(frontier)
            print("Picking random:", (x, y, z))
            return z * (H * W) + y * W + x

        return self.action_space.sample()
