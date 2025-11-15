import gymnasium as gym 
from gymnasium import spaces 
import numpy as np 
from minesweeper import Minesweeper

class MinesweeperEnv(gym.Env): 
    metadata = {"render_modes": ["ansi"], "render-fps": 4}

    def __init__(self, height=5, width=5, depth=5, num_mines=10, render_mode='ansi'): 
        super().__init__()

        self.height = height 
        self.width = width 
        self.depth = depth 
        self.num_mines = num_mines
        self.render_mode = render_mode

        self.game = None 

        self.observation_space = spaces.Box(low=-10, high=26, shape=(self.height, self.depth, self.width), dtype=np.int32)
        self.action_space = spaces.Discrete(self.height * self.width * self.depth)
        
    def _decode_action(self, action): 
        return divmod(action, self.height * self.width)[0], *divmod(divmod(action, self.height * self.width)[1], self.width)
        
    def reset(self, seed=None, options=None): 
        super().reset(seed=seed)

        self.game = Minesweeper(height=self.height, width=self.width, depth=self.depth, num_mines=self.num_mines, seed=seed)
        obs = self.game.get_observation()
        info = {}

        return obs, info 
    
    def step(self, action): 
        x, y, z = self._decode_action(action)
        prev_visible = np.sum(self.game.visible >= 0)

        if self.game.visible[x, y, z] == -2: 
            reward = -5.0
            terminated = False 
            truncated = False 
            obs = self.game.get_observation()
            return obs, reward, terminated, truncated, {} 
        
        if self.game.visible[x, y, z] >= 0:
            reward = -1.0
            terminated = False
            truncated = False
            obs = self.game.get_observation()
            return obs, reward, terminated, truncated, {}
        
        # Perform reveal
        self.game.reveal(x, y, z)
        self.game.update_surface_mask()
        obs = self.game.get_observation()

        # Game over handling
        if self.game.game_over:
            terminated = True
            truncated = False

            if self.game.win: reward = 100.0
            else: reward = -100.0

            return obs, reward, terminated, truncated, {}

        # Reward for progress: count newly revealed tiles
        new_visible = np.sum(self.game.visible >= 0)
        reward = float(new_visible - prev_visible)

        return obs, reward, False, False, {}
    
    def render_ansi(self):
        """
        Text-based 3D visualization of the board.
        Dimensions follow the convention:
            x = width
            y = height
            z = depth (layers)
        The observation obs[z, y, x] is printed one z-layer at a time.
        """
        obs = self.game.get_observation()
        out = []

        # Build each z-layer view
        for z in range(self.depth):
            out.append(f"Layer z={z}")
            for y in range(self.height):
                row = []
                for x in range(self.width):
                    v = obs[z, y, x]
                    if v == -2: row.append("█")        # buried/unexposed
                    elif v == -1: row.append("?")        # exposed but unknown
                    elif v == -3: row.append("*")        # revealed mine
                    else: row.append(str(v))     # revealed number 0–26
                out.append(" ".join(row))

        num_lines = self.depth + 1
        combined = [" | ".join(out[start + i] for start in range(0, len(out), num_lines)) for i in range(num_lines)]

        return "\n".join(combined)
    
    def render(self): 
        """
        Function that calls the corresponding render function. Currently only supports ANSI, but will in the future support 3D visualization using pyvista.
        """
        if self.render_mode == "ansi": return self.render_ansi()
    
if __name__ == "__main__": 
    print("This module is not meant to be run on its own! Please use ./run_agent.py instead.")