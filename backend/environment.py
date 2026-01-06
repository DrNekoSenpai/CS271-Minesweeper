import gymnasium as gym 
import numpy as np 
from gymnasium import spaces 
from backend.minesweeper import Minesweeper
from itertools import product

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
        self._plotter_off_screen = False

    def _init_plotter(self, off_screen: bool = False):
        import pyvista as pv
        self._plotter = pv.Plotter(off_screen=off_screen, window_size=(1920, 1080))
        self._plotter_off_screen = off_screen
        self._first_render = True

        self._plotter.add_axes()
        self._plotter.enable_eye_dome_lighting()
        self._plotter.set_background("black")
        # self._plotter.camera_position = "iso"
        
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
            reward = 10.0 if self.game.win else -10.0  # Scaled down rewards for stability
            safe_move = 1 if self.game.win else 0
            new_visible = np.sum(self.game.visible >= 0)
            safe_tiles = max(new_visible - prev_visible, 0)

            info["safe_move"] = safe_move
            info["safe_tiles"] = safe_tiles

            return obs, reward, terminated, truncated, info

        # Reward for progress
        new_visible = np.sum(self.game.visible >= 0)
        delta = int(new_visible - prev_visible)

        # Base reward: tiles revealed (normalized scale)
        reward = float(delta) * 0.5  # Much smaller multiplier to prevent explosion
        
        # Bonus for progress toward completion
        total_safe = self.width * self.height * self.depth - self.num_mines
        progress = new_visible / total_safe
        
        # Progress bonus (much more conservative)
        if delta > 0:
            reward += progress * 2.0  # Scaled down from 10.0
        
        # Bonus for high-value reveals (flood fills are very good)
        if delta > 5:
            reward += 1.0  # Scaled down from 5.0
        if delta > 10:
            reward += 2.0  # Scaled down from 10.0
        
        safe_tiles = max(delta, 0)

        # "Good move" semantics: only count if we actually revealed something new
        safe_move = 1 if safe_tiles > 0 else 0

        info["safe_move"] = safe_move
        info["safe_tiles"] = safe_tiles
        info["progress"] = progress

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

    def _render_pyvista(self, off_screen: bool = False):
        import pyvista as pv
        obs = self.game.get_observation()
        height, width, depth = obs.shape

        # Lazy init plotter (re-init if off_screen mode changes)
        if self._plotter is None or self._plotter_off_screen != off_screen:
            self._init_plotter(off_screen=off_screen)

        # Clear previous actors
        self._plotter.clear()

        cube_size = 1.0 
        half = cube_size / 2.0 

        # NEW: color map for numbers
        number_colors = {
            1: "dodgerblue",   # 1
            2: "limegreen",    # 2
            3: "yellow",       # 3
            4: "orange",       # 4
            5: "red",          # 5
            "6+": "magenta",   # 6 and above
        }

        # NEW: store points/labels per value bucket
        number_points_by_v = {k: [] for k in number_colors.keys()}
        number_labels_by_v = {k: [] for k in number_colors.keys()}

        for x, y, z in product(range(self.height), range(self.width), range(self.depth)):
            v = obs[x, y, z]

            wx = x * cube_size
            wy = y * cube_size
            wz = z * cube_size

            # Buried tiles cannot be seen
            if v == -2:
                continue

            # Exposed but unrevealed tiles are grey translucent cubes
            if v == -1:
                cube = pv.Cube(center=(wx, wy, wz), x_length=cube_size, y_length=cube_size, z_length=cube_size)
                self._plotter.add_mesh(cube, color="gray", opacity=0.2)
                continue

            # Revealed mines are red spheres
            if v == -10:
                sphere = pv.Sphere(radius=half * 0.7, center=(wx, wy, wz))
                self._plotter.add_mesh(sphere, color="red")

            # Tiles adjacent to mines (numbers), color-coded by value
            if v > 0: 
                bucket = v if v <= 5 else "6+"
                color = number_colors[bucket]

                cube = pv.Cube(
                    center=(wx, wy, wz),
                    x_length=cube_size,
                    y_length=cube_size,
                    z_length=cube_size,
                )
                # CHANGED: faint cube in per-number color
                self._plotter.add_mesh(cube, color=color, opacity=0)

                number_points_by_v[bucket].append([wx, wy, wz])
                number_labels_by_v[bucket].append(str(v))
                continue

        # NEW: add labels per bucket, with matching colors
        for bucket, pts_list in number_points_by_v.items():
            if not pts_list:
                continue

            pts = pv.PolyData(pts_list)
            labels = number_labels_by_v[bucket]
            color = number_colors[bucket]

            self._plotter.add_point_labels(
                pts,
                labels,
                font_size=32,
                text_color=color,
                point_size=0,
                shape=None,
                always_visible=True,
            )

        # Wireframe bounding box
        bounds = (
            -half, (height - 0.5) * cube_size,
            -half, (width - 0.5) * cube_size,
            -half, (depth - 0.5) * cube_size,
        )
        outline = pv.Cube(bounds=bounds)
        self._plotter.add_mesh(outline, style="wireframe", color="cyan", opacity=0.3)

        # Show / update
        if off_screen: self._plotter.render()
        else:
            if getattr(self, "_first_render", True):
                self._first_render = False
                self._plotter.show(auto_close=False)
            else:
                self._plotter.render()

        return None

    def render_frame(self, path: str, off_screen: bool = True):
        """
        Render a single 3D frame and optionally save to disk.
        """
        # Force 3D render logic
        self._render_pyvista(off_screen=off_screen)

        if path and self._plotter is not None:
            # PyVista will infer format from extension (.png recommended)
            self._plotter.screenshot(path)

        return None

    # --- update render() to call the new signature safely ---
    def render(self):
        if self.render_mode == "ansi": return self._render_ansi()
        elif self.render_mode == "3d": return self._render_pyvista(off_screen=False)
    
if __name__ == "__main__": 
    print("This module is not meant to be run on its own! Please use ./run_agent.py instead.")