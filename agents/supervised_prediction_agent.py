"""
Supervised Learning approach for Minesweeper
Predicts P(tile is safe) instead of Q-values

Key difference from DQN:
- No temporal credit assignment
- Direct classification: safe vs mine
- Trained on ground truth mine locations
- Simpler, more data efficient
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from agents.dueling_cnn_agent import Dueling3DCNN, ReplayBuffer


class MinePredictorNetwork(nn.Module):
    """
    Network that predicts P(tile is safe | board state)
    
    Uses same encoder as DQN but outputs probabilities instead of Q-values
    """
    def __init__(self, height: int, width: int, depth: int, hidden: int = 512):
        super().__init__()
        self.H, self.W, self.D = height, width, depth
        self.n_actions = height * width * depth
        
        # Reuse proven CNN architecture from DQN
        self.encoder = Dueling3DCNN(height, width, depth, hidden)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, D, H, W) board state
            
        Returns:
            probs: (B, n_actions) probability each tile is SAFE
        """
        # Use DQN encoder to get Q-values, treat as logits
        logits = self.encoder(x)  # (B, n_actions)
        
        # Sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        return probs


class SupervisedMinePredictor:
    """
    Agent that learns to identify safe tiles via supervised learning
    
    Key advantages over DQN:
    - No epsilon-greedy (always picks safest tile)
    - No replay buffer (can train batch-wise)
    - No target network (no temporal modeling)
    - Can use ground truth mine labels from completed games
    """
    
    def __init__(
        self,
        height: int,
        width: int,
        depth: int,
        lr: float = 1e-4,
        device: Optional[str] = None,
    ):
        self.H, self.W, self.D = height, width, depth
        self.n_actions = height * width * depth
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Network
        self.net = MinePredictorNetwork(height, width, depth).to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        
        print(f"SupervisedMinePredictor initialized on {self.device}")
    
    def select_action(self, obs: np.ndarray) -> int:
        """
        Select safest legal action (highest P(safe))
        
        Args:
            obs: (H, W, D) observation
            
        Returns:
            action: Index of safest legal tile
        """
        # Get legal actions
        legal = self._legal_action_indices(obs)
        if legal.size == 0:
            return np.random.randint(self.n_actions)
        
        # Get safety probabilities
        obs_b = np.expand_dims(obs, axis=0)
        x = self._obs_to_tensor(obs_b)
        
        with torch.no_grad():
            probs = self.net(x)[0]  # (n_actions,)
        
        # Mask to legal actions and pick safest
        probs_np = probs.cpu().numpy()
        legal_probs = probs_np[legal]
        best_legal_idx = np.argmax(legal_probs)
        
        return int(legal[best_legal_idx])
    
    def train_batch(self, boards: np.ndarray, mine_labels: np.ndarray, legal_masks: np.ndarray) -> dict:
        """
        Train on batch of boards with ground truth mine labels
        
        Args:
            boards: (B, H, W, D) board states
            mine_labels: (B, H, W, D) binary 1=mine, 0=safe
            legal_masks: (B, H, W, D) binary 1=legal action, 0=illegal
            
        Returns:
            stats: Dict with 'loss' and 'accuracy'
        """
        boards_t = self._obs_to_tensor(boards)  # (B, 3, D, H, W)
        
        # Flatten labels to match action space
        B = boards.shape[0]
        mine_labels_flat = mine_labels.reshape(B, -1)  # (B, n_actions)
        legal_masks_flat = legal_masks.reshape(B, -1)  # (B, n_actions)
        
        # Target: 1 if safe, 0 if mine
        targets = 1.0 - mine_labels_flat  # (B, n_actions)
        targets_t = torch.FloatTensor(targets).to(self.device)
        legal_t = torch.BoolTensor(legal_masks_flat).to(self.device)
        
        # Forward pass
        probs = self.net(boards_t)  # (B, n_actions)
        
        # Loss only on legal actions
        loss = F.binary_cross_entropy(probs[legal_t], targets_t[legal_t])
        
        # Backward pass
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        # Metrics
        with torch.no_grad():
            preds = (probs > 0.5).float()
            correct = (preds[legal_t] == targets_t[legal_t]).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': correct.item()
        }
    
    def _legal_action_indices(self, obs: np.ndarray) -> np.ndarray:
        """Get indices of legal actions (unrevealed exposed tiles)"""
        coords = np.argwhere(obs == -1)
        if coords.size == 0:
            return np.array([], dtype=np.int64)
        
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        indices = z * (self.H * self.W) + y * self.W + x
        return indices
    
    def _obs_to_tensor(self, obs_batch: np.ndarray) -> torch.Tensor:
        """Convert (B, H, W, D) to (B, 3, D, H, W) with augmentation"""
        # Same augmentation as DQN for consistency
        B = obs_batch.shape[0]
        augmented = []
        
        for i in range(B):
            obs = obs_batch[i]
            raw = obs.astype(np.float32) / 26.0
            safe_mask = self._compute_safe_mask(obs)
            danger_mask = np.zeros_like(obs, dtype=np.float32)
            danger_mask[(obs > 3) & (obs < 27)] = 1.0
            
            aug = np.stack([raw, safe_mask, danger_mask], axis=0)  # (3, H, W, D)
            augmented.append(aug)
        
        x = np.stack(augmented, axis=0)  # (B, 3, H, W, D)
        x = np.transpose(x, (0, 1, 4, 2, 3))  # (B, 3, D, H, W)
        return torch.from_numpy(x).to(self.device)
    
    def _compute_safe_mask(self, obs: np.ndarray) -> np.ndarray:
        """Compute mask of tiles adjacent to zeros (guaranteed safe)"""
        H, W, D = obs.shape
        mask = np.zeros((H, W, D), dtype=np.float32)
        zeros = np.argwhere(obs == 0)
        
        for zx, zy, zz in zeros:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == dy == dz == 0:
                            continue
                        nx, ny, nz = zx + dx, zy + dy, zz + dz
                        if 0 <= nx < H and 0 <= ny < W and 0 <= nz < D:
                            if obs[nx, ny, nz] == -1:
                                mask[nx, ny, nz] = 1.0
        return mask
    
    def save_checkpoint(self, path: str):
        """Save model weights"""
        torch.save({
            'net_state': self.net.state_dict(),
            'optim_state': self.optim.state_dict(),
        }, path)
        
    def load_checkpoint(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['net_state'])
        self.optim.load_state_dict(checkpoint['optim_state'])
