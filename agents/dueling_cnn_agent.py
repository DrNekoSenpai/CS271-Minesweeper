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

    Assumes input obs is (batch, 1, D, H, W).
    Output is Q-values for all actions (flattened voxels).
    """

    def __init__(self, height: int, width: int, depth: int, hidden: int = 512):
        super().__init__()
        self.H, self.W, self.D = height, width, depth
        self.n_actions = height * width * depth

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
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

        # Compute flattened dim with dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 1, depth, height, width)
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
        # x: (B,1,D,H,W)
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
    Double DQN + Dueling 3D CNN agent with action masking.

    Observation convention:
        obs shape = (H, W, D)
        Legal actions are tiles where obs[y, x, z] == -1 (exposed & unrevealed).

    Flattened action index convention:
        index = z * H * W + y * W + x
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
        target_update: int = 1_000,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 200_000,
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

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.online = Dueling3DCNN(height, width, depth, hidden=512).to(self.device)
        self.target = Dueling3DCNN(height, width, depth, hidden=512).to(self.device)

        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optim = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

    # -------------------------
    # Action masking helpers
    # -------------------------
    def legal_action_indices(self, obs: np.ndarray) -> np.ndarray:
        """
        obs: (H,W,D)
        returns flat indices of legal actions where obs == -1
        """
        coords = np.argwhere(obs == -1)  # (y,x,z)
        if coords.size == 0:
            return np.array([], dtype=np.int64)

        y = coords[:, 0]
        x = coords[:, 1]
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
    # Obs -> tensor
    # -------------------------
    def obs_to_tensor(self, obs_batch: np.ndarray) -> torch.Tensor:
        """
        Converts (B,H,W,D) to (B,1,D,H,W), float.
        """
        # Normalize lightly for stability
        x = obs_batch.astype(np.float32) / 26.0
        # (B,H,W,D) -> (B,D,H,W)
        x = np.transpose(x, (0, 3, 1, 2))
        x = torch.from_numpy(x).unsqueeze(1)  # (B,1,D,H,W)
        return x.to(self.device)

    # -------------------------
    # Single obs action
    # -------------------------
    def select_action(self, obs: np.ndarray) -> int:
        self.total_steps += 1
        eps = self.epsilon()

        legal = self.legal_action_indices(obs)
        if legal.size == 0:
            # fallback: allow anything (should be rare)
            return random.randrange(self.n_actions)

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
    def select_actions(self, obs_batch: np.ndarray) -> np.ndarray:
        """
        For vector envs.
        obs_batch shape: (B,H,W,D)
        returns actions shape: (B,)
        """
        B = obs_batch.shape[0]
        actions = np.zeros(B, dtype=np.int64)

        eps = self.epsilon()
        # Only increment once per batch call (keeps schedule sane)
        self.total_steps += B

        # Decide which envs explore
        explore_flags = np.random.rand(B) < eps

        # Build legal masks and handle empties
        legal_lists = [self.legal_action_indices(obs_batch[i]) for i in range(B)]

        # Exploration picks
        for i in range(B):
            legal = legal_lists[i]
            if legal.size == 0:
                actions[i] = random.randrange(self.n_actions)
            elif explore_flags[i]:
                actions[i] = int(random.choice(legal))

        # Exploitation for the rest
        exploit_idxs = [i for i in range(B) if not explore_flags[i] and legal_lists[i].size > 0]
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
        rewards_t = torch.from_numpy(rewards).to(self.device)
        done_t = torch.from_numpy(done).to(self.device)

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
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optim.step()

        # Update target periodically
        if self.total_steps % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())

        return {"loss": float(loss.item())}

    def save(self, path: str):
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optim": self.optim.state_dict(),
            "steps": self.total_steps,
            "H": self.H, "W": self.W, "D": self.D,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.optim.load_state_dict(ckpt["optim"])
        self.total_steps = ckpt.get("steps", 0)
