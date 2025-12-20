**File:** `agents/dueling_cnn_agent.py`
- Changed CNN input from 1 channel to 3 channels
- Added `augment_observation()` method (line 263):
  - Channel 0: Raw observation (normalized)
  - Channel 1: Safe mask (neighbors of revealed zeros)
  - Channel 2: Danger mask (high-probability mine locations)
- Added NaN/Inf safety checks for numerical stability
- Added `use_augmentation` parameter for debugging/performance testing
- **Status:** ✅ ENABLED (default: `use_augmentation=True`)

**File:** `pretrain_dqn_imitation.py` (new)
- Pre-train DQN agent using expert Bayesian agent trajectories
- Collects ~10k high-quality transitions from expert
- Performs 5000 supervised learning updates before RL training
- Added expert performance metrics (win rate, avg moves)
- Command: `python pretrain_dqn_imitation.py --size 5 --mines 5 --expert-episodes 500`
- **Status:** ✅ ENABLED (checkpoint loaded at training start)

**File:** `agents/dueling_cnn_agent.py`
- Added `_compute_safe_neighbors_mask()`: Identifies guaranteed safe tiles
- Added `get_safe_actions()`: Returns flat indices of safe moves
- Modified `select_action()` and `select_actions()` with `use_hybrid` parameter
- 30% chance to pick guaranteed safe move if available
- Falls back to DQN policy otherwise
- **Status:** ✅ ENABLED (`self.use_hybrid=True`)

**File:** `agents/bayesian_approximation_agent.py`
- Removed hardcoded `MINES=10`, `HEIGHT=5`, `WIDTH=5`, `DEPTH=5`
- Added parameters to `__init__()`: `height`, `width`, `depth`, `num_mines`
- Fixed probability calculations to use instance variables
- Fixed action encoding consistency

**File:** `agents/dueling_cnn_agent.py` (line 536)
- Added reward normalization (divide by 100) in `optimize()` method
- Prevents gradient explosion from large terminal rewards (±100)
- Normalized scale: wins=+1.0, losses=-1.0, progress=0.0-0.26
- **Impact:** Training loss stable instead of exploding

**File:** `pretrain_dqn_imitation.py`
- Fixed win/loss tracking to check `reward >= 100` instead of non-existent `info['result']`
- Added expert agent performance metrics display (win rate, avg moves, total transitions)

**File:** `train_dqn.py`
- Added architecture mismatch detection when loading checkpoints
- Gracefully handles old 1-channel vs new 3-channel model incompatibility
- Fixed checkpoint loading priority:
  1. Latest training checkpoint (`dqn-checkpoint-s{size}-m{mines}-{step}.pth`)
  2. Pretrained checkpoint (`dqn-pretrained-s{size}-m{mines}.pth`)
  3. Fallback to fresh start

**File:** `train_dqn.py`
- Fixed episode metrics to track last 100 completed episodes across ALL parallel envs
- Previously only logged single environment's metrics (incorrect rolling average)
- Added win count display: `wins={count}/100`
- Added timing metrics:
  - Progress: `step=X/Y`
  - Training speed: `steps/s`
  - Estimated time to completion: `ETA: X.Xh`
- Added device and config printout at training start

**File:** `train_dqn.py`
- Increased default `num-envs` from 8 to 16 (better GPU utilization)
- Increased default `batch-size` from 128 to 256 (better GPU utilization)
- Tested AsyncVectorEnv for parallel processing (reverted due to overhead)
- Tested training every N steps (reverted to every step for accuracy)
- **Final config:** SyncVectorEnv, train every step, 8-16 envs recommended

**File:** `train_dqn.py`
- `--size`: Board size (creates size×size×size cube)
- `--mines`: Number of mines
- `--num-steps`: Total training steps
- `--num-envs`: Parallel environments (higher = more GPU usage)
- `--batch-size`: Training batch size (higher = more GPU usage)
- `--lr`: Learning rate
- `--fresh`: Force fresh start (ignore checkpoints)