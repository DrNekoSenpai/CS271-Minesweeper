import numpy as np
from itertools import product
import random

# environment variables
HEIGHT = 5
WIDTH = 5
DEPTH = 5
MINES = 10

class Agent:
    """
    Bayesian Approximation Minesweeper agent:
    - Picks neighbors of revealed 0s first
    - Otherwise estimates mine probabilities and picks safest tile
    - Estimates probabilities based on adjacent revealed numbers
    - Uses bayesian approximation, basically this is a simple average
    - of the local probabilities derived from each adjacent number tile
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, obs):
        H, W, D = obs.shape
        zeros = np.argwhere(obs == 0)

        # check for 0 tiles for guaranteed safe moves
        for zx, zy, zz in zeros:
            for dx, dy, dz in product((-1,0,1), repeat=3):
                if dx == dy == dz == 0:
                    continue
                nx, ny, nz = zx + dx, zy + dy, zz + dz
                if 0 <= nx < H and 0 <= ny < W and 0 <= nz < D:
                    if obs[nx, ny, nz] == -1:
                        action = nz * (H * W) + ny * W + nx
                        # print("Picking safe:", (nx, ny, nz))
                        return action

        # base probability
        frontier = np.argwhere(obs == -1)
        base_probability = MINES / (HEIGHT * WIDTH * DEPTH)

        # estimate probabilities for each frontier tile
        probabilities = {}
        for x, y, z in frontier:
            local_probs = []

            for dx, dy, dz in product((-1,0,1), repeat=3):
                if dx == dy == dz == 0:
                    continue

                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < H and 0 <= ny < W and 0 <= nz < D:
                    if obs[nx, ny, nz] > 0:
                        number = obs[nx, ny, nz]

                        # count unrevealed neighbors of this number tile
                        unrevealed = 0
                        for ddx, ddy, ddz in product((-1,0,1), repeat=3):
                            if ddx == ddy == ddz == 0:
                                continue
                            ax, ay, az = nx + ddx, ny + ddy, nz + ddz
                            if 0 <= ax < H and 0 <= ay < W and 0 <= az < D:
                                if obs[ax, ay, az] == -1:
                                    unrevealed += 1

                        if unrevealed > 0:
                            local_probs.append(number / unrevealed)

            if local_probs:
                probabilities[(x, y, z)] = sum(local_probs) / len(local_probs)
            else:
                probabilities[(x, y, z)] = base_probability

        best_tile = min(probabilities, key=probabilities.get)
        x, y, z = best_tile
        # print("Picking probabilistic:", best_tile, "p =", probabilities[best_tile])

        return z * (H * W) + y * W + x