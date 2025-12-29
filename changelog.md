**File:** `agents/dueling_cnn_agent.py`
- Changed CNN input from 1 channel to 3 channels
- Added `augment_observation()` method (line 263):
  - Channel 0: Raw observation (normalized)
  - Channel 1: Safe mask (neighbors of revealed zeros)
  - Channel 2: Danger mask (high-probability mine locations)
- Added NaN/Inf safety checks for numerical stability
- Added `use_augmentation` parameter for debugging/performance testing

**File:** `pretrain_dqn_imitation.py` (new)
- Pre-train DQN agent using expert Bayesian agent trajectories
- Collects ~10k high-quality transitions from expert
- Performs 5000 supervised learning updates before RL training
- Added expert performance metrics (win rate, avg moves)
- Command: `python pretrain_dqn_imitation.py --size 5 --mines 5 --expert-episodes 500`

**File:** `agents/dueling_cnn_agent.py`
- Added `_compute_safe_neighbors_mask()`: Identifies guaranteed safe tiles
- Added `get_safe_actions()`: Returns flat indices of safe moves
- Modified `select_action()` and `select_actions()` with `use_hybrid` parameter
- 30% chance to pick guaranteed safe move if available
- Falls back to DQN policy otherwise

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

---

## DQN Learning Failure Fix (Dec 28, 2025)

**Problem Observed:**
- DQN showed 0% win rate after pretraining + 255k RL training steps
- Training logs showed 80-90% wins during training
- Evaluation with same config (hybrid mode enabled) showed only 4.2% wins
- Extensive debugging revealed no code bugs in win tracking or reward system

**Root Cause Analysis:**
1. **Pretraining contamination:** Original pretraining collected ALL episodes (wins + losses)
   - Bayesian expert only wins 27% of time
   - 70% of training data showed losing behavior
   - DQN learned conflicting patterns (both winning and losing strategies)

2. **Hybrid heuristic crutch:** Training used `use_hybrid=True` + `safe_action_prob=0.3`
   - Safe action heuristic (neighbors of zeros) available ~42% of time
   - When available, heuristic made correct moves 30% of the time
   - DQN only controlled 58-70% of actions, made poor choices
   - Training metrics (80-90% wins) were real but misleading - heuristic did the work
   - DQN never actually learned to play

3. **Insufficient pretraining:** Only 500 episodes = ~13k transitions
   - Model has millions of parameters
   - Not enough data for effective behavioral cloning

**Files Changed:**

**File:** `pretrain_dqn_imitation.py`
- **CRITICAL FIX:** Now collects N winning episodes instead of N mixed episodes
- Changed episode collection loop to discard all losing trajectories
- Only learns from successful expert behavior (pure signal, no noise)
- Updated metrics to show attempts needed vs winning episodes collected
- Changed win detection from `>= 100` to `== 100.0` (consistency with rest of codebase)
- **Impact:** DQN learns only correct strategies, no conflicting losing patterns

**File:** `agents/dueling_cnn_agent.py` (lines 188-194)
- **CRITICAL FIX:** Changed default `use_hybrid=False` and `safe_action_prob=0.0`
- Added comments explaining hybrid mode should be disabled during training
- Forces DQN to learn without safe-action heuristic crutch
- **Impact:** DQN must actually learn to play, can't rely on heuristic doing the work

**File:** `train_dqn.py` (line 49)
- Added comment explaining hybrid mode is disabled by agent defaults
- Clarifies that DQN is forced to learn without heuristic assistance
- Hybrid mode can still be enabled in evaluate_dqn.py for performance boost

**Expected Outcomes:**
- Pretrained model should show >10% win rate before RL training
- Training metrics will be lower initially (no heuristic boost) but reflect true learning
- Evaluation without hybrid should match training performance
- Final model with hybrid enabled should exceed pure heuristic performance

**Testing Changes:**
Created comprehensive test suite (see `run_all_tests.py`):
- `test_win_tracking.py`: Validates win/loss detection consistency
- `test_hybrid_mode.py`: Measures hybrid action selection rates
- `test_safe_detection.py`: Tests safe action availability
- `test_safe_legal.py`: Validates safe action correctness
- `test_action_encoding.py`: Tests action space encoding
- `debug_training_wins.py`: Simulates actual training loop
- `diagnose_dqn.py`: Compares Q-values between models

**New Automation:**
- `run_all_tests.py`: Runs entire test suite in sequence
- `run_full_training_pipeline.py`: Pretrains then trains automatically