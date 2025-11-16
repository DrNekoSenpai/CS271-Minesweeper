import numpy as np
from itertools import product, combinations
from collections import defaultdict, deque

class Agent:
    """
    bayesian agent that uses constraint satisfaction and Bayesian inference
    to select the safest action based on the current observation.
    1) Detect forced moves (safe or mine) from revealed tiles.
    2) Build constraints from revealed tiles.
    3) Cluster frontier tiles into independent groups.
    4) For each group, compute posterior mine probabilities via enumeration.
    5) If any tile has P=0 (safe), select it immediately.
    6) Otherwise, choose the tile with the lowest mine probability,
       breaking ties by distance to center.

    more or less what happens here is that we're using posterior probabilities
    to determine moves in the state space. the primary problem with this approach
    is that the combinatorial explosion of possible mine configurations makes
    it infeasible to compute exact probabilities for large groups of tiles, so there
    is a heuristic that targets small independent clusters of tiles towards the
    center, and will fall back to a center-guess heuristic if the cluster is too large.
    Furthermore this agent immediately selects any tile that is guaranteed safe (P=0).
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, obs):
        D, H, W = obs.shape

        frontier = np.argwhere(obs == -1)
        if len(frontier) == 0:
            return self.action_space.sample()

        revealed = np.argwhere(obs >= 0)

        # see helper below
        forced_safe, forced_mine = self.find_forced_moves(obs, revealed)
        if forced_safe:
            z, y, x = forced_safe[0]
            return z * H * W + y * W + x

        # see helper below
        constraints = self.build_constraints(obs, revealed)

        # without any constraints we choose the center-most tile
        if not constraints:
            return self.pick_center_guess(frontier, D, H, W)

        # building clusters here for efficient calculation, see helper
        clusters = self.build_clusters(frontier, constraints)

        # posteriors are bult for each cluster, see helper
        post = {}
        for group, c_list in clusters:
            local_post = self.compute_local_posteriors(group, c_list)
            post.update(local_post)

        # IMMEDIATE EXIT - select the first P=0 tile found
        zero_prob = [c for c, p in post.items() if abs(p) < 1e-12]
        if zero_prob:
            z, y, x = zero_prob[0]
            return z * H * W + y * W + x

        # otherwise pick lowest probability ties
        # tiebreaker is to choose the tile closest to center
        # using cartesian distance
        best = None
        best_score = (float("inf"), float("inf"))

        for (z, y, x) in post:
            p = post[(z, y, x)]
            dist = ((z - (D/2))**2 + (y - (H/2))**2 + (x - (W/2))**2)
            score = (p, dist)
            if score < best_score:
                best_score = score
                best = (z, y, x)

        z, y, x = best
        return z * H * W + y * W + x

    # the reason we need forced moves is that they can simplify the problem
    # significantly by reducing the number of unknowns
    # I AM ACTUALLY UNSURE IF WE NEED THIS AS OF RIGHT NOW, BUT AS FAR
    # AS TESTING GOES, IT SEEMS TO HELP A BIT
    def find_forced_moves(self, obs, revealed):
        D, H, W = obs.shape
        forced_safe = []
        forced_mine = []

        for (z, y, x) in revealed:
            val = obs[z, y, x]
            neigh = self.get_neighbors(z, y, x, obs)

            unrevealed = [(nz, ny, nx) for (nz, ny, nx) in neigh if obs[nz, ny, nx] == -1]
            if not unrevealed:
                continue

            # count known mines (flags) around
            known_mines = 0 

            # required mines to satisfy this tile
            req = val - known_mines

            # case A: all unrevealed must be safe
            if req == 0:
                forced_safe.extend(unrevealed)

            # case B: all unrevealed must be mines
            if req == len(unrevealed):
                forced_mine.extend(unrevealed)

        return forced_safe, forced_mine

    # this helper builds constraints from revealed tiles
    def build_constraints(self, obs, revealed):
        constraints = []
        for (z, y, x) in revealed:
            val = obs[z, y, x]
            neigh = self.get_neighbors(z, y, x, obs)
            frontier_cells = [(nz, ny, nx) for (nz, ny, nx) in neigh if obs[nz, ny, nx] == -1]
            if frontier_cells:
                constraints.append((tuple(frontier_cells), val))
        return constraints

    # this helper builds clusters of connected frontier tiles
    def build_clusters(self, frontier, constraints):
        """
        Build graph where frontier tiles are nodes.
        Two tiles connect if they appear in the same constraint.
        Return list of (group_nodes, constraints_in_group).
        """
        # we need tuples for hashing
        # so we convert here
        frontier = [tuple(f) for f in frontier]

        graph = defaultdict(list)

        # graph is built by connecting all tiles in each constraint
        for cells, _ in constraints:
            for a in cells:
                for b in cells:
                    if a != b:
                        graph[a].append(b)

        # simple breadth-first search to find components
        visited = set()
        clusters = []

        for f in frontier:
            if f in visited:
                continue

            q = deque([f])
            visited.add(f)
            comp = [f]

            while q:
                u = q.popleft()
                for v in graph[u]:
                    if v not in visited:
                        visited.add(v)
                        q.append(v)
                        comp.append(v)

            # generate constraints for this component
            comp_set = set(comp)
            comp_constraints = []
            for cells, val in constraints:
                if any(c in comp_set for c in cells):
                    comp_constraints.append((cells, val))

            clusters.append((comp, comp_constraints))

        return clusters

    # this helper computes posterior probabilities for a small group
    def compute_local_posteriors(self, group, constraints):
        """
        Enumerate only this small group of tiles (<= 12 recommended).
        """
        group = [tuple(g) for g in group]
        n = len(group)

        # INTERRUPT if cluster is too large
        # otherwise combinatorial explosion
        if n > 12:
            return {c: 0.5 for c in group}

        index = {c: i for i, c in enumerate(group)}
        total_valid = 0
        mine_count = {c: 0 for c in group}

        for assign in product((0, 1), repeat=n):
            if not self.check_constraints(assign, group, index, constraints):
                continue

            total_valid += 1
            for c in group:
                if assign[index[c]] == 1:
                    mine_count[c] += 1

        if total_valid == 0:
            return {c: 0.5 for c in group}

        return {c: mine_count[c] / total_valid for c in group}

    # this helper checks if a given assignment satisfies all constraints
    # the constraints are in the form of (cells, required_mines)
    def check_constraints(self, assign, group, index, constraints):
        for cells, required in constraints:
            s = 0
            for c in cells:
                if c in index:
                    s += assign[index[c]]
            if s != required:
                return False
        return True

    # this helper picks the center-most guess from the frontier
    def pick_center_guess(self, frontier, D, H, W):
        best = None
        best_dist = float("inf")
        for (z, y, x) in frontier:
            dist = ( (z - (D/2))**2 + (y - (H/2))**2 + (x - (W/2))**2 )
            if dist < best_dist:
                best_dist = dist
                best = (z, y, x)
        z, y, x = best
        return z * H * W + y * W + x

    # this helper gets all valid neighbors of a cell
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
