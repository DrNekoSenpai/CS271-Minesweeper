import numpy as np
from itertools import product

class Agent:
    """
    this is just a simple heuristic agent for minesweeper
    it looks for tiles adjacent to revealed 0s to pick safe moves first,
    then scores other tiles based on the average of their revealed neighbors
    and picks the one with the lowest score (least likely to be a mine)
    if no info is available, it picks randomly from the frontier
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, obs):
        D, H, W = obs.shape
        frontier = np.argwhere(obs == -1)
        if len(frontier) == 0:
            print("guessing randomly, no frontier tiles left")
            return self.action_space.sample()

        revealed = np.argwhere(obs >= 0)

        # pick 0 tiles first
        safe_candidates = []
        for z, y, x in frontier:
            neighbors = self.get_neighbors(z, y, x, obs)
            if any(obs[nz, ny, nx] == 0 for nz, ny, nx in neighbors):
                safe_candidates.append((z, y, x))
        if safe_candidates:
            z, y, x = safe_candidates[0]
            print("guaranteed safe move found (adjacent to 0)")
            return z * H * W + y * W + x

        # simple heuristic: score tiles by average of revealed neighbors
        best_tile = None
        best_score = float("inf")
        for z, y, x in frontier:
            neighbors = self.get_neighbors(z, y, x, obs)
            # ignore unrevealed neighbors in score
            numbers = [obs[nz, ny, nx] for nz, ny, nx in neighbors if obs[nz, ny, nx] >= 0]
            if numbers:
                score = np.mean(numbers)
            else:
                score = 0  # no info, treat as low risk
            # tiebreaker: distance to center
            dist = ((z - D/2)**2 + (y - H/2)**2 + (x - W/2)**2)
            if (score, dist) < (best_score, 0):
                best_score = score
                best_tile = (z, y, x)

        if best_tile:
            z, y, x = best_tile
            print(f"Picking tile with heuristic score {best_score:.2f}")
            return z * H * W + y * W + x

        # fallback: pick random
        z, y, x = frontier[0]
        print("Fallback: picking random frontier tile")
        return z * H * W + y * W + x

    def get_neighbors(self, z, y, x, obs):
        D, H, W = obs.shape
        res = []
        for dz, dy, dx in product((-1, 0, 1), repeat=3):
            if dz == dy == dx == 0:
                continue
            nz, ny, nx = z + dz, y + dy, x + dx
            if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                res.append((nz, ny, nx))
        return res
