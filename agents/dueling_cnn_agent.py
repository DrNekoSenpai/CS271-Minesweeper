import random
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

class ResBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.c1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.c2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.n1 = nn.GroupNorm(8, channels)
        self.n2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        identity = x
        x = F.relu(self.n1(self.c1(x)))
        x = self.n2(self.c2(x))
        return F.relu(x + identity)
    
# -------------------------
# 3D Dueling CNN Network
# -------------------------
class Dueling3DCNN(nn.Module):
    """
    Dueling 3D CNN for Minesweeper.

    Assumes input obs is (batch, 3, D, H, W) - now supports multi-channel input.
    Channel 0: Raw observation
    Channel 1: Safe neighbors mask (neighbors of zeros)
    Channel 2: Danger mask (high numbers indicating potential mines)
    Output is Q-values for all actions (flattened voxels).
    """

    def __init__(self, height: int, width: int, depth: int, hidden: int = 512, in_channels: int = 3):
        super().__init__()
        self.H, self.W, self.D = height, width, depth
        self.n_actions = height * width * depth

        # Stem - now accepts 3 channels
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
        )

        # Stage 1
        self.s1 = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            ResBlock3D(64),
            ResBlock3D(64),
        )

        # Stage 2
        self.s2 = nn.Sequential(
            nn.Conv3d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            ResBlock3D(128),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, depth, height, width)
            f = self._forward_trunk(dummy)
            flat_dim = f.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat_dim, hidden),
            nn.ReLU(),
        )

        # Dueling heads
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

        self.adv = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, self.n_actions),
        )

    def _forward_trunk(self, x):
        x = self.stem(x)
        x = self.s1(x)
        x = self.s2(x)
        return x

    def forward(self, x):
        # x: (B,3,D,H,W) - now expects 3 channels
        x = self._forward_trunk(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        v = self.value(x)  # (B,1)
        a = self.adv(x)    # (B,n_actions)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q

# -------------------------
# Replay Buffer
# -------------------------
@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool

class ReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append(Transition(obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs = np.stack([t.obs for t in batch], axis=0)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_obs = np.stack([t.next_obs for t in batch], axis=0)
        done = np.array([t.done for t in batch], dtype=np.float32)
        return obs, actions, rewards, next_obs, done

    def __len__(self):
        return len(self.buffer)


# -------------------------
# Agent
# -------------------------
class DuelingDCNNAgent:
    """
    Double DQN + Dueling 3D CNN agent with action masking and state augmentation.

    Observation convention:
        obs shape = (H, W, D)
        Legal actions are tiles where obs[x, y, z] == -1 (exposed & unrevealed).

    Flattened action index convention:
        index = z * H * W + y * W + x
    
    New features:
        - Multi-channel state representation (raw + safe mask + danger mask)
        - Hybrid action selection (mix DQN with guaranteed safe moves)
        - Support for imitation learning bootstrap
    """

    def __init__(
        self,
        height: int,
        width: int,
        depth: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_size: int = 200_000,
        warmup: int = 2_000,
        target_update: int = 250,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 400_000,
        grad_clip: float = 1.0,
        device: Optional[str] = None,
    ):
        self.H, self.W, self.D = height, width, depth
        self.n_actions = height * width * depth

        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup = warmup
        self.target_update = target_update
        self.grad_clip = grad_clip

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.total_steps = 0
        
        # Hybrid action selection parameters
        # IMPORTANT: Set to False during training to force DQN to learn!
        # Only enable during evaluation if you want the heuristic boost
        self.use_hybrid = False
        self.safe_action_prob = 0.0  # Disabled during training

        if device is None:
            # Check if CUDA is truly usable (not just available)
            if torch.cuda.is_available():
                try:
                    # Test if CUDA actually works
                    torch.zeros(1).cuda()
                    device = "cuda"
                except RuntimeError:
                    # CUDA available but not working (e.g., unsupported GPU)
                    device = "cpu"
                    print("Warning: CUDA detected but not functional. Using CPU instead.")
            else:
                device = "cpu"
        self.device = torch.device(device)

        self.online = Dueling3DCNN(height, width, depth, hidden=512, in_channels=3).to(self.device)
        self.target = Dueling3DCNN(height, width, depth, hidden=512, in_channels=3).to(self.device)

        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optim = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        
        # Mixed precision training (automatic if CUDA available)
        self.use_amp = (self.device.type == 'cuda')
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        if self.use_amp:
            print(f"Mixed precision training enabled (FP16) - expect ~2x speedup")

    # -------------------------
    # Action masking helpers
    # -------------------------
    def legal_action_indices(self, obs: np.ndarray) -> np.ndarray:
        """
        obs: (H,W,D)
        returns flat indices of legal actions where obs == -1
        """
        coords = np.argwhere(obs == -1)  # (x, y, z)
        if coords.size == 0:
            return np.array([], dtype=np.int64)

        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]

        idx = z * (self.H * self.W) + y * self.W + x
        return idx.astype(np.int64)

    def build_legal_mask(self, obs_batch: np.ndarray) -> torch.Tensor:
        """
        obs_batch: (B,H,W,D)
        returns mask: (B, n_actions) boolean tensor
        """
        B = obs_batch.shape[0]
        mask = np.zeros((B, self.n_actions), dtype=np.bool_)

        for i in range(B):
            idxs = self.legal_action_indices(obs_batch[i])
            if idxs.size > 0:
                mask[i, idxs] = True

        return torch.from_numpy(mask).to(self.device)

    # -------------------------
    # Epsilon schedule
    # -------------------------
    def epsilon(self) -> float:
        t = min(self.total_steps, self.eps_decay_steps)
        frac = t / float(self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    # -------------------------
    # State Augmentation (#7: Multi-channel representation)
    # -------------------------
    def augment_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Augment single observation with domain knowledge channels.
        
        Args:
            obs: (H, W, D) raw observation
            
        Returns:
            augmented: (3, H, W, D) with channels:
                [0] = raw observation (normalized)
                [1] = safe neighbors mask (neighbors of zeros)
                [2] = danger mask (high adjacent mine counts)
        """
        H, W, D = obs.shape
        
        # Channel 0: Normalized raw observation
        raw = obs.astype(np.float32) / 26.0
        
        # Channel 1: Safe neighbors mask (tiles adjacent to zeros)
        safe_mask = self._compute_safe_neighbors_mask(obs)
        
        # Channel 2: Danger mask (tiles with high numbers nearby)
        danger_mask = np.zeros_like(obs, dtype=np.float32)
        danger_mask[(obs > 3) & (obs < 27)] = 1.0  # High numbers indicate danger
        
        # Stack channels: (3, H, W, D)
        augmented = np.stack([raw, safe_mask, danger_mask], axis=0)
        
        # Safety check for NaN/Inf
        if not np.isfinite(augmented).all():
            print(f"WARNING: Non-finite values in augmented observation!")
            print(f"Raw range: [{raw.min()}, {raw.max()}]")
            print(f"Safe mask range: [{safe_mask.min()}, {safe_mask.max()}]")
            print(f"Danger mask range: [{danger_mask.min()}, {danger_mask.max()}]")
            augmented = np.nan_to_num(augmented, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return augmented
    
    def _compute_safe_neighbors_mask(self, obs: np.ndarray) -> np.ndarray:
        """
        Create mask marking tiles adjacent to revealed zeros (guaranteed safe).
        
        Args:
            obs: (H, W, D) observation
            
        Returns:
            mask: (H, W, D) binary mask where 1.0 = safe neighbor of zero
        """
        H, W, D = obs.shape
        mask = np.zeros((H, W, D), dtype=np.float32)
        
        # Find all zero tiles
        zeros = np.argwhere(obs == 0)
        
        # Mark all unrevealed neighbors of zeros as safe
        for zx, zy, zz in zeros:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == dy == dz == 0:
                            continue
                        nx, ny, nz = zx + dx, zy + dy, zz + dz
                        if 0 <= nx < H and 0 <= ny < W and 0 <= nz < D:
                            if obs[nx, ny, nz] == -1:  # Unrevealed
                                mask[nx, ny, nz] = 1.0
        
        return mask

    # -------------------------
    # Obs -> tensor
    # -------------------------
    def obs_to_tensor(self, obs_batch: np.ndarray, use_augmentation: bool = True) -> torch.Tensor:
        """
        Converts (B,H,W,D) to (B,3,D,H,W) with augmented channels.
        """
        B = obs_batch.shape[0]
        
        if not use_augmentation:
            # Simple 1-channel fallback for debugging
            x = obs_batch.astype(np.float32) / 26.0
            x = np.expand_dims(x, axis=1)  # (B, 1, H, W, D)
            x = np.transpose(x, (0, 1, 4, 2, 3))  # (B, 1, D, H, W)
            # Repeat to 3 channels
            x = np.repeat(x, 3, axis=1)
            return torch.from_numpy(x).to(self.device)
        
        augmented_batch = []
        
        for i in range(B):
            aug = self.augment_observation(obs_batch[i])  # (3, H, W, D)
            augmented_batch.append(aug)
        
        # Stack: (B, 3, H, W, D)
        x = np.stack(augmented_batch, axis=0)
        # Transpose: (B, 3, H, W, D) -> (B, 3, D, H, W)
        x = np.transpose(x, (0, 1, 4, 2, 3))
        x = torch.from_numpy(x)
        return x.to(self.device)

    # -------------------------
    # Safe Action Detection (#8: Hybrid approach)
    # -------------------------
    def get_safe_actions(self, obs: np.ndarray) -> np.ndarray:
        """
        Get guaranteed safe actions (neighbors of revealed zeros).
        
        Args:
            obs: (H, W, D) observation
            
        Returns:
            safe_actions: flat indices of guaranteed safe moves
        """
        H, W, D = obs.shape
        safe_coords = []
        
        # Find all zero tiles
        zeros = np.argwhere(obs == 0)
        
        # Check all neighbors of zeros
        for zx, zy, zz in zeros:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == dy == dz == 0:
                            continue
                        nx, ny, nz = zx + dx, zy + dy, zz + dz
                        if 0 <= nx < H and 0 <= ny < W and 0 <= nz < D:
                            if obs[nx, ny, nz] == -1:  # Unrevealed
                                safe_coords.append((nx, ny, nz))
        
        if not safe_coords:
            return np.array([], dtype=np.int64)
        
        # Remove duplicates and convert to flat indices
        safe_coords = list(set(safe_coords))
        safe_actions = []
        for x, y, z in safe_coords:
            idx = z * (H * W) + y * W + x
            safe_actions.append(idx)
        
        return np.array(safe_actions, dtype=np.int64)

    # -------------------------
    # Single obs action
    # -------------------------
    def select_action(self, obs: np.ndarray, use_hybrid: bool = None) -> int:
        """
        Select action with optional hybrid approach.
        
        Args:
            obs: (H, W, D) observation
            use_hybrid: Override self.use_hybrid if specified
        """
        self.total_steps += 1
        eps = self.epsilon()
        
        if use_hybrid is None:
            use_hybrid = self.use_hybrid

        legal = self.legal_action_indices(obs)
        if legal.size == 0:
            # fallback: allow anything (should be rare)
            return random.randrange(self.n_actions)
        
        # Hybrid: Sometimes take guaranteed safe moves (#8)
        if use_hybrid:
            safe_actions = self.get_safe_actions(obs)
            if safe_actions.size > 0 and random.random() < self.safe_action_prob:
                return int(random.choice(safe_actions))

        # Explore
        if random.random() < eps:
            return int(random.choice(legal))

        # Exploit with mask
        obs_b = np.expand_dims(obs, axis=0)  # (1,H,W,D)
        x = self.obs_to_tensor(obs_b)
        with torch.no_grad():
            q = self.online(x)[0]  # (n_actions,)

            mask = torch.zeros(self.n_actions, dtype=torch.bool, device=self.device)
            mask[torch.from_numpy(legal).to(self.device)] = True

            q_masked = q.clone()
            q_masked[~mask] = -1e9
            action = int(torch.argmax(q_masked).item())

        return action

    # -------------------------
    # Parallel obs action
    # -------------------------
    def select_actions(self, obs_batch: np.ndarray, use_hybrid: bool = None) -> np.ndarray:
        """
        For vector envs with optional hybrid action selection.
        obs_batch shape: (B,H,W,D)
        returns actions shape: (B,)
        """
        B = obs_batch.shape[0]
        actions = np.zeros(B, dtype=np.int64)

        eps = self.epsilon()
        # Only increment once per batch call (keeps schedule sane)
        self.total_steps += B
        
        if use_hybrid is None:
            use_hybrid = self.use_hybrid

        # Decide which envs explore
        explore_flags = np.random.rand(B) < eps
        
        # Hybrid: Check for safe actions first (#8)
        safe_flags = np.zeros(B, dtype=bool)
        if use_hybrid:
            for i in range(B):
                safe_actions = self.get_safe_actions(obs_batch[i])
                if safe_actions.size > 0 and random.random() < self.safe_action_prob:
                    actions[i] = int(random.choice(safe_actions))
                    safe_flags[i] = True

        # Build legal masks and handle empties
        legal_lists = [self.legal_action_indices(obs_batch[i]) for i in range(B)]

        # Exploration picks (skip if already handled by safe action)
        for i in range(B):
            if safe_flags[i]:
                continue
            legal = legal_lists[i]
            if legal.size == 0:
                actions[i] = random.randrange(self.n_actions)
            elif explore_flags[i]:
                actions[i] = int(random.choice(legal))

        # Exploitation for the rest (skip safe and explore)
        exploit_idxs = [i for i in range(B) if not safe_flags[i] and not explore_flags[i] and legal_lists[i].size > 0]
        if exploit_idxs:
            sub_obs = obs_batch[exploit_idxs]
            x = self.obs_to_tensor(sub_obs)

            with torch.no_grad():
                q = self.online(x)  # (b, n_actions)

            # Apply legality mask per sample
            legal_mask = self.build_legal_mask(sub_obs)  # (b, n_actions)
            q = q.masked_fill(~legal_mask, -1e9)
            best = torch.argmax(q, dim=1).cpu().numpy()

            for j, env_i in enumerate(exploit_idxs):
                actions[env_i] = best[j]

        return actions

    # -------------------------
    # Learning
    # -------------------------
    def remember(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)

    def optimize(self) -> Dict[str, float]:
        if len(self.buffer) < max(self.warmup, self.batch_size):
            return {}

        obs, actions, rewards, next_obs, done = self.buffer.sample(self.batch_size)

        obs_t = self.obs_to_tensor(obs)
        next_obs_t = self.obs_to_tensor(next_obs)

        actions_t = torch.from_numpy(actions).to(self.device).unsqueeze(1)
        # Clip rewards to [-10, 10] range for stability
        rewards_clipped = np.clip(rewards, -10.0, 10.0)
        rewards_t = torch.from_numpy(rewards_clipped).to(self.device)
        done_t = torch.from_numpy(done).to(self.device)

        # Mixed precision training context
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # Current Q(s,a)
            q = self.online(obs_t).gather(1, actions_t).squeeze(1)

            # -------- Double DQN with legality masking --------
            with torch.no_grad():
                # Online chooses next action among LEGAL moves
                online_next_q = self.online(next_obs_t)  # (B, n_actions)
                legal_mask = self.build_legal_mask(next_obs)  # (B, n_actions)

                online_next_q = online_next_q.masked_fill(~legal_mask, -1e9)
                next_actions = torch.argmax(online_next_q, dim=1, keepdim=True)

                # Target evaluates those actions
                target_next_q = self.target(next_obs_t).gather(1, next_actions).squeeze(1)

                target = rewards_t + (1.0 - done_t) * self.gamma * target_next_q

            loss = F.smooth_l1_loss(q, target)

        self.optim.zero_grad()
        
        # Scaled backward pass for mixed precision
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Track gradient norm before clipping
        grad_norm = 0.0
        for p in self.online.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        # Gradient clipping
        if self.use_amp:
            self.scaler.unscale_(self.optim)
        nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        
        # Optimizer step
        if self.use_amp:
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            self.optim.step()

        # Update target periodically
        if self.total_steps % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())

        return {
            "loss": float(loss.item()),
            "grad_norm": float(grad_norm),
            "q_mean": float(q.mean().item()),
            "target_mean": float(target.mean().item())
        }

    def save_checkpoint(self, path: str, step: int) -> None:
        """
        Save training state so we can resume later.

        Args:
            path: filesystem path to save to (e.g. 'checkpoint_step_50000.pth')
            step: current global training step in the outer loop
        """
        checkpoint = {
            "step": step,
            "total_steps": self.total_steps,
            "online_state": self.online.state_dict(),
            "target_state": self.target.state_dict(),
            "optimizer_state": self.optim.state_dict(),
            # Hyperparameters (optional but nice for safety / auditing)
            "H": self.H,
            "W": self.W,
            "D": self.D,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "warmup": self.warmup,
            "target_update": self.target_update,
            "eps_start": self.eps_start,
            "eps_end": self.eps_end,
            "eps_decay_steps": self.eps_decay_steps,
        }

        # If you ever want to also save the replay buffer, you can do:
        # checkpoint["replay_buffer"] = list(self.buffer.buffer)

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> int:
        """
        Load training state from a checkpoint.

        Returns:
            step (int): the global training step to resume from
                        (use this to set your training loop start).
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.online.load_state_dict(checkpoint["online_state"])
        self.target.load_state_dict(checkpoint["target_state"])
        self.optim.load_state_dict(checkpoint["optimizer_state"])

        # Restore step counters so epsilon schedule & target updates line up
        self.total_steps = checkpoint.get("total_steps", 0)
        step = checkpoint.get("step", self.total_steps)

        # If you chose to save replay buffer:
        # if "replay_buffer" in checkpoint:
        #     from collections import deque
        #     self.buffer.buffer = deque(
        #         checkpoint["replay_buffer"],
        #         maxlen=self.buffer.buffer.maxlen
        #     )

        # Make sure target is in eval mode (just to be explicit)
        self.target.eval()

        return step
    
    # -------------------------
    # Inference save/load
    # -------------------------
    def save_model(self, path: str) -> None:
        """
        Save only what you need to run inference later.
        """
        payload = {
            "online_state": self.online.state_dict(),
            "H": self.H,
            "W": self.W,
            "D": self.D,
        }
        torch.save(payload, path)

    def load_model(self, path: str) -> None:
        """
        Load inference weights.
        """
        payload = torch.load(path, map_location=self.device)

        # Support both styles:
        # 1) our payload dict
        # 2) raw state_dict
        state = payload["online_state"] if isinstance(payload, dict) and "online_state" in payload else payload

        self.online.load_state_dict(state)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
