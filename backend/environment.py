import gymnasium as gym 
from gymnasium import spaces 
import numpy as np 
from backend.minesweeper import Minesweeper
from itertools import product

# -- todo -- NOT AS IMPORTANT FOR NOW
# 3D rendering? although i dont know if this is required for the final report
# as long as we can guarantee the correctness of the gamestate at each turn
# it might be easier to just write unit testing for this.

class MinesweeperEnv(gym.Env): 
    metadata = {"render_modes": ["ansi", "3d"], "render-fps": 4}

    def __init__(self, height=5, width=5, depth=5, num_mines=10, render_mode='3d'): 
        super().__init__()

        self.height = height 
        self.width = width 
        self.depth = depth 
        self.num_mines = num_mines
        self.render_mode = render_mode

        self.game = None 

        self.observation_space = spaces.Box(low=-10, high=26, shape=(self.height, self.depth, self.width), dtype=np.int32)
        self.action_space = spaces.Discrete(self.height * self.width * self.depth)

        # PyVista-related
        self._plotter = None 
        self._first_render = True
        
    def _decode_action(self, action): 
        z, rem = divmod(action, self.height * self.width)
        y, x = divmod(rem, self.width)
        return z, y, x
        
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
    
    def _render_ansi(self):
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
                    if v == -2: row.append("█")   # buried/unexposed
                    elif v == -1: row.append("?") # exposed but unknown
                    elif v == -10: row.append("*") # revealed mine
                    else: row.append(str(v))      # revealed number 0–26
                out.append(" ".join(row))

        num_lines = self.depth + 1
        combined = [" | ".join(out[start + i] for start in range(0, len(out), num_lines)) for i in range(num_lines)]

        return "\n".join(combined)
    
    def _render_pyvista(self): 
        import pyvista as pv
        obs = self.game.get_observation()
        height, width, depth = obs.shape

        # Lazy init plotter
        if self._plotter is None: 
            self._plotter = pv.Plotter()
            self._plotter.add_axes()
            self._plotter.enable_eye_dome_lighting()
            self._plotter.set_background("black")

        # Clear previous actors 
        self._plotter.clear()

        cube_size = 1.0 
        half = cube_size / 2.0 

        number_points = []
        number_labels = []
        mine_points = []

        for x, y, z in product(range(self.height), range(self.width), range(self.depth)): 
            v = obs[y, x, z] 

            wx = x * cube_size 
            wy = y * cube_size 
            wz = z * cube_size 

            # Buried tiles cannot be seen
            if v == -2: continue 

            # Exposed but unrevealed tiles are grey translucent cubes
            if v == -1: 
                cube = pv.Cube(center=(wx, wy, wz), x_length=cube_size, y_length=cube_size, z_length=cube_size)
                self._plotter.add_mesh(cube, color="gray", opacity=0.2)
                continue

            # Revealed mines are red spheres
            if v == -10: 
                sphere = pv.Sphere(radius=half*0.7, center=(wx,wy,wz))
                self._plotter.add_mesh(sphere, color="red")
                # Continue 

            # Tiles adjacent to mines are faint cubes with number labels. Completely safe tiles are also not shown
            if v > 0: 
                cube = pv.Cube(center=(wx, wy, wz), x_length=cube_size, y_length=cube_size, z_length=cube_size)
                self._plotter.add_mesh(cube, color="white", opacity=0.08)

                number_points.append([wx, wy, wz])
                number_labels.append(str(v))

        if number_points: 
            pts = pv.PolyData(number_points)
            self._plotter.add_point_labels(pts, number_labels, font_size=16, text_color="white", point_size=0, shape=None, always_visible=True)

        # Optional: wireframe bounding box for the whole cube
        # Bounds are in (xmin, xmax, ymin, ymax, zmin, zmax)
        bounds = (
            -half, (height - 0.5) * cube_size,
            -half, (width - 0.5) * cube_size,
            -half, (depth - 0.5) * cube_size,
        )
        outline = pv.Cube(bounds=bounds)
        self._plotter.add_mesh(outline, style="wireframe", color="cyan", opacity=0.3)

        # Show / update
        if getattr(self, "_first_render", True):
            self._first_render = False
            self._plotter.show(auto_close=False)  # keep window open
        else:
            self._plotter.render()

        # For Gymnasium's "3d" mode, returning None is fine
        return None
    
    def render(self): 
        """
        Function that calls the corresponding render function. Currently only supports ANSI, but will in the future support 3D visualization using pyvista.
        """
        if self.render_mode == "ansi": return self._render_ansi()
        elif self.render_mode == "3d": return self._render_pyvista()
    
if __name__ == "__main__": 
    print("This module is not meant to be run on its own! Please use ./run_agent.py instead.")