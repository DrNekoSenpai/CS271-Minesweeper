# CHANGELOG

## Core Improvements

**3-Channel State Augmentation**
- Changed CNN input from 1→3 channels: raw observation, safe mask (neighbors of zeros), danger mask (high-prob mines). Provides richer spatial context for better decision-making.

**Imitation Learning Pre-training**
- New `pretrain_dqn_imitation.py`: Pre-trains DQN with ~10k expert Bayesian trajectories (5000 supervised updates). Bootstraps agent with strong starting policy.

**Hybrid Action Selection**
- Added safe move detection with 30% probability to select guaranteed safe tiles when available. Reduces random exploration deaths while maintaining DQN exploration.

## Critical Bug Fixes

**Bayesian Agent Hardcoded Parameters**
- Removed hardcoded `MINES=10`, `HEIGHT/WIDTH/DEPTH=5` from `agents/bayesian_approximation_agent.py`. Fixed loss explosion when pre-training with different configurations.

**Reward Normalization**
- Divide rewards by 100 in `agents/dueling_cnn_agent.py` (line 536). Scales ±100 terminal rewards to ±1.0, prevents gradient explosion during training.

**Architecture Mismatch Handling**
- Added checkpoint loading error detection in `train_dqn.py`. Gracefully handles old 1-channel vs new 3-channel model incompatibility with fallback to fresh start.

**Episode Metrics Tracking**
- Fixed `train_dqn.py` to track last 100 episodes across ALL parallel environments. Previously only logged single env (incorrect rolling average).

## Training Infrastructure

**Complete Metrics Logging**
- Log format: `[step=X/Y] reward=Z safe_moves=A safe_tiles=B wins=C/100`. Includes all 5 metrics + timing info (steps/s, ETA). Backward compatible with old formats.

**Win Rate Tracking & Graphing**
- Added win rate calculation and dedicated graph (0-100% y-axis). Shows wins per 100 episodes alongside reward/loss curves for better progress visibility.

**Milestone Checkpoint Preservation**
- Keep checkpoints at 100k intervals (modulo check), delete intermediate 5k saves. Preserves important training milestones while managing disk space.

**Train Every 4 Steps Optimization**
- Training only occurs every 4 steps instead of every step. With 8 envs, collects 32 samples per update. Provides 3-4x speedup with minimal quality loss.

## CLI Parameters

**Key Arguments:**
- `--size`: Board dimension (creates size³ cube)
- `--mines`: Mine count
- `--num-steps`: Total training steps
- `--num-envs`: Parallel environments (default: 16)
- `--batch-size`: Training batch size (default: 256)
- `--lr`: Learning rate (default: 1e-4)
- `--fresh`: Ignore existing checkpoints, start fresh