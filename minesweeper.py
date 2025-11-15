import numpy as np
from collections import deque
from itertools import product

class Minesweeper:
    def __init__(self, height=5, width=5, depth=5, num_mines=10, seed=None): 
        self.height = height 
        self.width = width 
        self.depth = depth 
        self.num_mines = num_mines
        self.rng = np.random.default_rng(seed) 

        self.new_game()

    def new_game(self): 
        """
        Reset the Minesweeper board and generate a new game state. Initializes both the internal board and the visible board. Mines are randomly placed according to the configured mine count, and adjacency values are computed for every tile. The visible board is set so that all tiles start in a hidden state.

        Returns: (np.ndarray) A copy of the initial visible board representing the observation returned to the agent. Hidden tiles are encoded as -1.
        """
        self.visible = np.full((self.height, self.width, self.depth), -2, dtype=int)
        
        # Expose all six sides of the cube so that the agent / player can only click on the top layer, decreasing complexity 
        self.visible[0, :, :]  = -1
        self.visible[-1, :, :] = -1
        self.visible[:, 0, :]  = -1
        self.visible[:, -1, :] = -1
        self.visible[:, :, 0]  = -1
        self.visible[:, :, -1] = -1

        self.board = np.zeros((self.height, self.width, self.depth), dtype=int)
        mine_positions = self.rng.choice(self.depth * self.height * self.width, self.num_mines, replace=False)

        for mine in mine_positions: 
            x, y, z = divmod(mine, self.height * self.width)[0], *divmod(divmod(mine, self.height * self.width)[1], self.width)

            # -1 is mine
            self.board[x, y, z] = -1

        for x, y, z in product(range(self.height), range(self.width), range(self.depth)): 
            if self.board[x, y, z] == -1: continue 
            self.board[x, y, z] = self._count_adjacent_mines(x, y, z)

        self.game_over = False 
        self.win = False 
        return self.get_observation()

    def reveal(self, x, y, z): 
        """
        Reveal a tile at the specified coordinates, applying standard Minesweeper logic. 
        - If the tile contains a mine, the agent loses and the game enters a terminal losing state. 
        - If the tile is a numbered tile, it is revealed normally. 
        - If the tile contains a zero, a flood reveal is triggered to reveal all connected zero-valued tiles and their border numbers. 

        Params: x, y, z (ints); tile height, width, depth respectively.
        """
        if self.game_over: return 
        if self.visible[x, y, z] != -1: return 

        # Case for hitting a mine 
        if self.board[x, y, z] == -1:
            self.visible[x, y, z] = -10
            self.game_over = True 
            self.win = False 
            return

        # Reveal all adjacent tiles if applicable
        self._flood_reveal(x, y, z)
        self.update_surface_mask()

        if np.sum(self.visible != -1) == (self.width * self.height * self.depth - self.num_mines): 
            self.game_over = True 
            self.win = True 

    def _flood_reveal(self, x, y, z): 
        """
        Performs a flood reveal starting from a zero-valued tile, recursively revealing all neighboring tiles that are also zero-valued, along with their bordering numbered tiles. This process uses a queue to avoid recursion depth issues. Only tiles that are in a hidden state are eligible to be revealed. 

        Params: x, y, z (ints); height, width, depth of the starting tile respectively.
        """
        queue = deque([(x, y, z)])
        while queue: 
            dqx, dqy, dqz = queue.popleft()
            if self.visible[dqx, dqy, dqz] != -1: continue 

            self.visible[dqx, dqy, dqz] = self.board[dqx, dqy, dqz]

            # If 0, reveal neighbors 
            if self.board[dqx, dqy, dqz] == 0: 
                directions = [(dx, dy, dz) for dx, dy, dz in product((-1, 0, 1), repeat=3) if not (dx == dy == dz == 0)]
                for dx, dy, dz in directions: 
                    nx = x + dx; ny = y + dy; nz = z + dz 
                    if 0 <= nx < self.height and 0 <= ny < self.width and 0 <= nz < self.depth: 
                        if self.visible[nx, ny, nz] == -1: 
                            queue.append((nx, ny, nz))
    
    def get_observation(self): 
        """
        Returns current observable gamestate, which is what a RL agent or human would be able to see during play. Hidden tiles remain encoded as -1, and revealed tiles show their corresponding numerical value. If a mine is revealed, that tile is encoded using a special negative value. 

        Returns: (np.ndarray) A copy of the visible board. 
        """
        return np.copy(self.visible)

    def _count_adjacent_mines(self, x, y, z): 
        """
        Count the number of mines adjacent to a given tile. In a D-dimensional Minesweeper board, a tile has up to 3^D - 1 neighbors. This method checks all valid neighboring coordinates and counts the number of mines adjacent to it. 

        Params: x, y, z (ints); tile height, width, depth respectively. 
        Returns: (int) Number of adjacent mines. Returns 0 if no neighbors contain mines. 
        """
        directions = [(dx, dy, dz) for dx, dy, dz in product((-1, 0, 1), repeat=3) if not (dx == dy == dz == 0)]
        count = 0

        for dx, dy, dz in directions: 
            nx = x + dx; ny = y + dy; nz = z + dz
            if 0 <= nx < self.height and 0 <= ny < self.width and 0 <= nz < self.depth: 
                if self.board[nx, ny, nz] == -1: count += 1

        return count 
    
    def update_surface_mask(self): 
        """
        Classifies all unrevealed tiles as exposed or buried based on their adjacency to revealed space. An unrevealed tile is considered "exposed" if at least one of its six face-adjacent neighbors has been revealed, meaning it is on the current outer surface of the carved three-dimensional Minesweeper volume.  

        Value conventions: 
          -1 : exposed (legal action target) 
          -2 : buried  (illegal action target) 
        """
        for x, y, z in product(range(self.height), range(self.width), range(self.depth)): 
            if self.visible[x, y, z] not in [-1, -2]: continue 

            if x in (0, self.height-1) or y in (0, self.width-1) or z in (0, self.depth-1): 
                self.visible[x, y, z] = -1
                continue

            exposed = False 

            for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]: 
                nx, ny, nz = x + dx, y + dy, z + dz 

                if 0 <= nx < self.height and 0 <= ny < self.width and 0 <= nz < self.depth: 
                    if self.visible[nx, ny, nz] >= 0: exposed = True; break 

            self.visible[x, y, z] = -1 if exposed else -2 

if __name__ == "__main__": 
    print("This module is not meant to be run on its own! Please use ./run_agent.py instead.")