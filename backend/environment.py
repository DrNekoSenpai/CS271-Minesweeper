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

    def __init__(self, height=5, width=5, depth=5, num_mines=15, render_mode='3d'): 
        super().__init__()

        self.height = height 
        self.width = width 
        self.depth = depth 
        self.num_mines = num_mines
        self.render_mode = render_mode

        self.game = None 

        self.observation_space = spaces.Box(low=-10, high=26, shape=(self.height, self.width, self.depth), dtype=np.int32)
        self.action_space = spaces.Discrete(self.height * self.width * self.depth)

        # PyVista-related
        self._plotter = None 
        self._first_render = True
        
    def _decode_action(self, action): 
        z, rem = divmod(action, self.height * self.width)
        y, x = divmod(rem, self.width)
        return x, y, z
        
    def reset(self, seed=None, options=None): 
        super().reset(seed=seed)

        self.game = Minesweeper(height=self.height, width=self.width, depth=self.depth, num_mines=self.num_mines, seed=seed)
        obs = self.game.get_observation()
        info = {}

        return obs, info 
    
    def step(self, action):
        # Get current obs and legal mask
        obs = self.game.get_observation()
        mask = self.get_action_mask(obs)
        legal = np.flatnonzero(mask)

        info = {}
        safe_move = 0 

        # Rebound if illegal
        if action < 0 or action >= self.action_space.n or not mask[action]:
            if legal.size == 0:
                # No legal actions available (rare edge case)
                # Just return state unchanged with neutral reward
                return obs, 0.0, False, False, {"no_legal_actions": True, "safe_move": 0, "safe_tiles": 0}

            rebound_action = int(np.random.choice(legal))
            info["illegal_action"] = int(action)
            info["rebound_action"] = rebound_action
            action = rebound_action

        # Decode using consistent convention
        x, y, z = self._decode_action(action)

        prev_visible = np.sum(self.game.visible >= 0)

        # Perform reveal
        self.game.reveal(x, y, z)
        self.game.update_surface_mask()
        obs = self.game.get_observation()

        # Terminal handling
        if self.game.game_over:
            terminated = True
            truncated = False
            reward = 100.0 if self.game.win else -100.0
            safe_move = 1 if self.game.win else 0
            new_visible = np.sum(self.game.visible >= 0)
            safe_tiles = max(new_visible - prev_visible, 0)

            info["safe_move"] = safe_move
            info["safe_tiles"] = safe_tiles

            return obs, reward, terminated, truncated, info

        # Reward for progress
        new_visible = np.sum(self.game.visible >= 0)
        delta = int(new_visible - prev_visible)

        reward = float(delta)
        safe_tiles = max(delta, 0)

        # "Good move" semantics: only count if we actually revealed something new
        safe_move = 1 if safe_tiles > 0 else 0

        info["safe_move"] = safe_move
        info["safe_tiles"] = safe_tiles

        return obs, reward, False, False, info

    def get_action_mask(self, obs=None):
        """
        Returns a boolean mask of size H*W*D.
        True = legal action (exposed & unrevealed).
        """
        if obs is None:
            obs = self.game.get_observation()

        H, W, D = obs.shape
        mask = np.zeros(H * W * D, dtype=bool)

        coords = np.argwhere(obs == -1)
        if coords.size == 0:
            return mask

        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]

        idx = z * (H * W) + y * W + x
        mask[idx] = True
        return mask

    def _render_ansi(self):
        obs = self.game.get_observation()  # (H, W, D)
        out = []

        for z in range(self.depth):
            out.append(f"Layer z={z}")
            for y in range(self.height):
                row = []
                for x in range(self.width):
                    v = obs[x, y, z]
                    if v == -2:
                        row.append("█")
                    elif v == -1:
                        row.append("?")
                    elif v == -10:
                        row.append("*")
                    else:
                        row.append(str(v))
                out.append(" ".join(row))
            out.append("")

        return "\n".join(out)
    
    def _render_pyvista(self):
        import pyvista as pv
        from itertools import product

        obs = self.game.get_observation()
        H, W, D = obs.shape

        # Lazy init plotter
        if getattr(self, "_plotter", None) is None:
            self._plotter = pv.Plotter(window_size=(1280, 720))
            self._plotter.set_background("black")
            self._plotter.enable_eye_dome_lighting()
            self._plotter.add_axes()

            # Stable camera for video (isometric-style)
            # You can also set a custom position tuple if you prefer.
            self._plotter.camera_position = "iso"
            self._plotter.camera.zoom(1.1)

            self._first_render = True

        # Remove previous actors while preserving camera
        self._plotter.clear()

        cube_size = 1.0
        half = cube_size / 2.0

        # Collect points by type
        unrevealed_pts = []
        number_pts = []
        number_labels = []
        mine_pts = []

        # Use obs dimensions, not self.height/width/depth just in case
        for x, y, z in product(range(H), range(W), range(D)):
            v = obs[x, y, z]

            # World coords
            wx = x * cube_size
            wy = y * cube_size
            wz = z * cube_size

            # Buried tiles cannot be seen
            if v == -2:
                continue

            # Exposed but unrevealed
            if v == -1:
                unrevealed_pts.append([wx, wy, wz])
                continue

            # Revealed mine
            if v == -10:
                mine_pts.append([wx, wy, wz])
                continue

            # Revealed number
            if v > 0:
                number_pts.append([wx, wy, wz])
                number_labels.append(str(v))
                continue

            # v == 0: you can choose to show or hide
            # For visual cleanliness, we hide zeros.

        # ---------- Draw unrevealed cubes (glyphs) ----------
        if unrevealed_pts:
            pts = pv.PolyData(unrevealed_pts)
            cube = pv.Cube(x_length=cube_size, y_length=cube_size, z_length=cube_size)
            cubes = pts.glyph(geom=cube, scale=False)
            self._plotter.add_mesh(cubes, color="gray", opacity=0.22)

        # ---------- Draw numbered cubes (faint) ----------
        if number_pts:
            pts = pv.PolyData(number_pts)
            cube = pv.Cube(x_length=cube_size, y_length=cube_size, z_length=cube_size)
            cubes = pts.glyph(geom=cube, scale=False)
            self._plotter.add_mesh(cubes, color="white", opacity=0.08)

            # Labels
            self._plotter.add_point_labels(
                pts,
                number_labels,
                font_size=16,
                text_color="white",
                point_size=0,
                shape=None,
                always_visible=True
            )

        # ---------- Draw mines as spheres ----------
        if mine_pts:
            pts = pv.PolyData(mine_pts)
            sphere = pv.Sphere(radius=half * 0.7)
            spheres = pts.glyph(geom=sphere, scale=False)
            self._plotter.add_mesh(spheres, color="red")

        # ---------- Outline bounding box ----------
        bounds = (
            -half, (H - 0.5) * cube_size,
            -half, (W - 0.5) * cube_size,
            -half, (D - 0.5) * cube_size,
        )
        outline = pv.Cube(bounds=bounds)
        self._plotter.add_mesh(outline, style="wireframe", color="cyan", opacity=0.3)

        # Show / update
        if getattr(self, "_first_render", True):
            self._first_render = False
            self._plotter.show(auto_close=False)
        else:
            self._plotter.render()

        return None
    
    def render(self): 
        """
        Function that calls the corresponding render function. Currently only supports ANSI, but will in the future support 3D visualization using pyvista.
        """
        if self.render_mode == "ansi": return self._render_ansi()
        elif self.render_mode == "3d": return self._render_pyvista()
    
if __name__ == "__main__": 
    print("This module is not meant to be run on its own! Please use ./run_agent.py instead.")