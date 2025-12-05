"""
Deep Q-Learning agent (dueling-style) for the Minesweeper environment.

This file provides an Agent class compatible with the existing
`run_agent.py` loop. `run_agent.py` only calls `select_action(obs)`, so
the agent implements that for inference. In addition the class exposes
methods to collect experience (`remember`) and train (`optimize` /
`save` / `load`) so you can train the agent from a separate training
script or REPL.

Notes:
- This implementation uses PyTorch. Add `torch` to your environment.
- `select_action(obs)` expects the observation array shape (D, H, W)
  and returns a flat index compatible with `run_agent.decode_action`.
"""

import collections
import random
from typing import Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer:
	def __init__(self, capacity: int):
		self.buffer = collections.deque(maxlen=capacity)

	def push(self, state, action, reward, next_state, done):
		self.buffer.append((state, action, reward, next_state, done))

	def sample(self, batch_size: int):
		batch = random.sample(self.buffer, batch_size)
		states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
		return states, actions, rewards, next_states, dones

	def __len__(self):
		return len(self.buffer)


class DuelingDQN(nn.Module):
	def __init__(self, in_channels: int, D: int, H: int, W: int, n_actions: int):
		super().__init__()

		# Small 3D conv trunk. Input shape (N, C, D, H, W)
		self.conv = nn.Sequential(
			nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv3d(32, 64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.AdaptiveAvgPool3d((1, 1, 1)),
		)

		# After adaptive avg pool we have a flat vector
		self.feature_dim = 64

		# Dueling streams: value and advantage
		self.value_stream = nn.Sequential(
			nn.Linear(self.feature_dim, 128),
			nn.ReLU(),
			nn.Linear(128, 1),
		)

		self.adv_stream = nn.Sequential(
			nn.Linear(self.feature_dim, 128),
			nn.ReLU(),
			nn.Linear(128, n_actions),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (N, C, D, H, W)
		features = self.conv(x)
		features = features.view(features.size(0), -1)
		value = self.value_stream(features)
		adv = self.adv_stream(features)
		# Q = V + (A - mean(A))
		q = value + (adv - adv.mean(dim=1, keepdim=True))
		return q


class Agent:
	"""DQN agent with a simple replay buffer and dueling conv network.

	Usage notes:
	- `select_action(obs)` returns a single integer action (flat index).
	- During training you should call `remember(state, action, reward, next_state, done)`
	  after each step, and periodically call `optimize()` to update the network.
	"""

	def __init__(self, action_space,
				 device: str = None,
				 replay_capacity: int = 10000,
				 batch_size: int = 64,
				 gamma: float = 0.99,
				 lr: float = 1e-3,
				 eps_start: float = 1.0,
				 eps_final: float = 0.05,
				 eps_decay: int = 10000,
				 target_update: int = 500,
				 in_channels: int = 1,
				 seed: int = 0):

		self.action_space = action_space
		self.n_actions = int(action_space.n)
		self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
		self.replay = ReplayBuffer(replay_capacity)
		self.batch_size = batch_size
		self.gamma = gamma
		self.lr = lr
		self.eps_start = eps_start
		self.eps_final = eps_final
		self.eps_decay = eps_decay
		self.frame_idx = 0
		self.target_update = target_update

		# We will lazily build networks when we see the first observation to infer shapes
		self.policy_net = None
		self.target_net = None
		self.optimizer = None

		torch.manual_seed(seed)
		random.seed(seed)
		np.random.seed(seed)

	def _build_networks(self, obs_shape: Tuple[int, int, int]):
		# obs_shape: (D, H, W)
		D, H, W = obs_shape
		in_channels = 1
		self.policy_net = DuelingDQN(in_channels, D, H, W, self.n_actions).to(self.device)
		self.target_net = DuelingDQN(in_channels, D, H, W, self.n_actions).to(self.device)
		self.target_net.load_state_dict(self.policy_net.state_dict())
		self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

	def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
		# obs shape: (D, H, W) with integer encodings (hidden/revealed/mine)
		# Normalize to small floats; use single channel
		arr = obs.astype(np.float32) / 10.0  # keep values small; -10 -> -1.0
		# shape to (1, C, D, H, W)
		t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(self.device)
		return t

	def _legal_actions(self, obs: np.ndarray):
		"""Return a numpy array of flat action indices that are legal (obs == -1)."""
		D, H, W = obs.shape
		coords = np.argwhere(obs == -1)
		if coords.size == 0:
			return np.array([], dtype=np.int64)
		# coords are (z, y, x)
		idxs = coords[:, 0] * H * W + coords[:, 1] * W + coords[:, 2]
		return idxs.astype(np.int64)

	def _legal_mask_tensor(self, obs: np.ndarray) -> torch.Tensor:
		"""Return a boolean mask tensor on the agent device of shape (n_actions,) marking legal actions."""
		mask = torch.zeros(self.n_actions, dtype=torch.bool, device=self.device)
		legal = self._legal_actions(obs)
		if legal.size > 0:
			idx_t = torch.from_numpy(legal).long().to(self.device)
			mask.index_fill_(0, idx_t, True)
		return mask

	def select_action(self, obs: np.ndarray) -> int:
		"""Return an action index for the observation using epsilon-greedy policy.

		This keeps compatibility with `run_agent.py` which expects a flat integer index.
		"""
		if self.policy_net is None:
			# lazy init when first observation is seen
			self._build_networks(obs.shape)

		self.frame_idx += 1
		eps = self.eps_final + (self.eps_start - self.eps_final) * np.exp(-1.0 * self.frame_idx / self.eps_decay)

		if random.random() < eps:
			# exploration: pick uniformly from legal (exposed) tiles if available
			legal = self._legal_actions(obs)
			if legal.size == 0:
				# fallback to action_space if no legal actions found
				return int(self.action_space.sample())
			return int(np.random.choice(legal))

		with torch.no_grad():
			t = self._obs_to_tensor(obs)
			qvals = self.policy_net(t).squeeze(0)  # shape (n_actions,)
			legal_mask = self._legal_mask_tensor(obs)
			if legal_mask.sum().item() == 0:
				# no known legal actions; fallback to argmax over all
				return int(qvals.argmax().item())
			# mask out illegal actions with a large negative value so argmax ignores them
			masked = qvals.clone()
			masked[~legal_mask] = -1e8
			action = int(masked.argmax().item())
			return action

	def remember(self, state, action, reward, next_state, done):
		# store raw numpy arrays (or convertible types)
		self.replay.push(state, int(action), float(reward), next_state if next_state is None else np.copy(next_state), bool(done))

	def optimize(self, num_iters: int = 1):
		"""Run optimization steps. Call periodically during training."""
		if self.policy_net is None or len(self.replay) < self.batch_size:
			return

		for _ in range(num_iters):
			states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

			# convert to tensors
			state_t = torch.from_numpy(states.astype(np.float32) / 10.0).unsqueeze(1).to(self.device)
			next_state_t = torch.from_numpy(next_states.astype(np.float32) / 10.0).unsqueeze(1).to(self.device)
			action_t = torch.from_numpy(actions).long().to(self.device)
			reward_t = torch.from_numpy(rewards).float().to(self.device)
			done_t = torch.from_numpy(dones.astype(np.uint8)).float().to(self.device)

			q_values = self.policy_net(state_t)
			q_value = q_values.gather(1, action_t.unsqueeze(1)).squeeze(1)

			with torch.no_grad():
				next_q_values = self.target_net(next_state_t)
				next_q_value = next_q_values.max(1)[0]
				expected_q = reward_t + self.gamma * next_q_value * (1.0 - done_t)

			loss = F.mse_loss(q_value, expected_q)

			self.optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
			self.optimizer.step()

			# soft update target periodically
			if self.frame_idx % self.target_update == 0:
				self.target_net.load_state_dict(self.policy_net.state_dict())

	def save(self, path: str):
		state = {
			"policy": self.policy_net.state_dict() if self.policy_net is not None else None,
			"optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
			"frame_idx": self.frame_idx,
		}
		torch.save(state, path)

	def load(self, path: str):
		data = torch.load(path, map_location=self.device)
		if data.get("policy") is None:
			return
		if self.policy_net is None:
			# user must provide an observation-shape-compatible network; assume small cube 5x5x5 fallback
			self._build_networks((5, 5, 5))
		self.policy_net.load_state_dict(data["policy"]) 
		self.target_net.load_state_dict(data["policy"]) 
		if data.get("optimizer") is not None and self.optimizer is not None:
			self.optimizer.load_state_dict(data["optimizer"]) 
		self.frame_idx = int(data.get("frame_idx", 0))

