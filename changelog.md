**File:** `agents/dueling_cnn_agent.py`
- Changed CNN input from 1 channel to 3 channels
- Added `augment_observation()` method:
  - Channel 0: Raw observation
  - Channel 1: Safe mask (neighbors of revealed zeros)
  - Channel 2: Danger mask (high-probability mine locations)
- Added NaN/Inf safety checks for numerical stability
- Added `use_augmentation` parameter for debugging

#### 2. Imitation Learning Bootstrap (#2)
**Files:** `pretrain_dqn_imitation.py` (new)
- Pre-train DQN agent using expert Bayesian agent trajectories
- Fills replay buffer with ~10k high-quality transitions
- Performs 5000 supervised learning updates before RL training
- Command: `python pretrain_dqn_imitation.py --size 5 --mines 5`

#### 3. Hybrid Action Selection (#8)
**File:** `agents/dueling_cnn_agent.py`
- Added `_compute_safe_neighbors_mask()`: Identifies guaranteed safe tiles
- Added `get_safe_actions()`: Returns flat indices of safe moves
- Modified `select_action()` with `use_hybrid` parameter
- 30% chance to pick guaranteed safe move if available
- Falls back to DQN policy otherwise

### Bug Fixes

#### Bayesian Agent Parameter Fix
**File:** `agents/bayesian_approximation_agent.py`
- Removed hardcoded `MINES=10`, `HEIGHT=5`, `WIDTH=5`, `DEPTH=5`
- Added parameters to `__init__()`: `height`, `width`, `depth`, `num_mines`
- Fixed probability calculations to use instance variables

#### Pre-training Metrics
**File:** `pretrain_dqn_imitation.py`
- Fixed win/loss tracking to check `reward >= 100` instead of non-existent `info['result']`
- Added expert agent performance metrics display

### Environment Setup

#### Python Version
- Switched from Python 3.14 to 3.11 for PyVista/VTK compatibility

#### GPU Support
- Added CPU fallback for RTX 5070 (Blackwell sm_120 architecture unsupported)
- Implemented device detection with automatic fallback

#### Command-Line Arguments
**File:** `train_dqn.py`
- Added `--size`, `--mines`, `--num-steps` arguments
- Simplified main() to accept CLI parameters

### Usage

#### Pre-training
```bash
python pretrain_dqn_imitation.py --size 5 --mines 5 --expert-episodes 500 --lr 5e-5
```

#### Training
```bash
python train_dqn.py --size 5 --mines 5 --num-steps 250000
```

#### Evaluation
```bash
python run_dqn_agent.py --checkpoint dqn-checkpoint-s5-m5-250000.pth --episodes 100
```
