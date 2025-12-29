"""
Full training pipeline: Pretraining (imitation) → RL Training (DQN)
Automates the two-stage training process so you don't need to manually run both commands.

Stage 1: Pretraining (Imitation Learning)
    - Collects winning episodes from Bayesian expert
    - Trains DQN via behavioral cloning
    - Saves pretrained checkpoint

Stage 2: RL Training (Deep Q-Learning)
    - Loads pretrained checkpoint
    - Continues training with experience replay
    - Saves checkpoints every 100k steps

Usage:
    # Default: 5x5x5 board, 8 mines, 10k winning episodes, 300k RL steps
    python run_full_training_pipeline.py
    
    # Custom configuration
    python run_full_training_pipeline.py --size 5 --mines 8 --expert-episodes 10000 --rl-steps 300000
    
    # Quick test run (fewer episodes/steps)
    python run_full_training_pipeline.py --expert-episodes 1000 --rl-steps 50000

Arguments:
    --size: Board size (creates size×size×size cube) [default: 5]
    --mines: Number of mines [default: 8]
    --expert-episodes: Number of WINNING episodes to collect [default: 10000]
    --num-updates: Number of pretraining optimization updates [default: 100000]
    --rl-steps: Number of RL training steps [default: 300000]
    --num-envs: Parallel environments for RL training [default: 8]
    --skip-pretrain: Skip pretraining if checkpoint already exists
"""

import subprocess
import sys
import argparse
import time
from pathlib import Path

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'


def print_header(text):
    """Print a formatted header."""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")


def run_command(cmd, description):
    """Run a command and stream output in real-time."""
    print_header(description)
    print(f"{CYAN}Command: {' '.join(cmd)}{RESET}\n")
    
    start_time = time.time()
    
    try:
        # Get project root and set PYTHONPATH
        project_root = Path(__file__).parent.absolute()
        import os
        env = os.environ.copy()
        env['PYTHONPATH'] = str(project_root)
        
        # Run command with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            cwd=str(project_root)
        )
        
        # Stream output line by line
        for line in process.stdout:
            print(line, end='')
        
        # Wait for completion
        process.wait()
        elapsed = time.time() - start_time
        
        if process.returncode == 0:
            print(f"\n{GREEN}[PASS] {description} completed successfully{RESET} ({elapsed:.1f}s)")
            return True
        else:
            print(f"\n{RED}[FAIL] {description} failed{RESET} (exit code: {process.returncode}, {elapsed:.1f}s)")
            return False
            
    except Exception as e:
        print(f"\n{RED}[ERROR] Error running {description}: {str(e)}{RESET}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Full training pipeline: Pretraining → RL Training',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Configuration arguments
    parser.add_argument('--size', type=int, default=5,
                        help='Board size (creates size×size×size cube) [default: 5]')
    parser.add_argument('--mines', type=int, default=5,
                        help='Number of mines [default: 5]')
    parser.add_argument('--expert-episodes', type=int, default=10000,
                        help='Number of WINNING episodes to collect for pretraining [default: 10000]')
    parser.add_argument('--num-updates', type=int, default=100000,
                        help='Number of pretraining optimization updates [default: 100000]')
    parser.add_argument('--rl-steps', type=int, default=300000,
                        help='Number of RL training steps [default: 300000]')
    parser.add_argument('--num-envs', type=int, default=8,
                        help='Parallel environments for RL training [default: 8]')
    parser.add_argument('--skip-pretrain', action='store_true',
                        help='Skip pretraining if checkpoint already exists')
    
    args = parser.parse_args()
    
    # Print configuration
    print_header("3D MINESWEEPER DQN - FULL TRAINING PIPELINE")
    print(f"{CYAN}Configuration:{RESET}")
    print(f"  Board size:        {args.size}×{args.size}×{args.size}")
    print(f"  Mines:             {args.mines}")
    print(f"  Expert episodes:   {args.expert_episodes} (winning only)")
    print(f"  Pretraining updates: {args.num_updates}")
    print(f"  RL training steps: {args.rl_steps}")
    print(f"  Parallel envs:     {args.num_envs}")
    
    # Check for existing pretrained checkpoint
    pretrained_checkpoint = f"dqn-pretrained-s{args.size}-m{args.mines}.pth"
    checkpoint_exists = Path(pretrained_checkpoint).exists()
    
    if checkpoint_exists and args.skip_pretrain:
        print(f"\n{YELLOW}[SKIP] Skipping pretraining - checkpoint exists: {pretrained_checkpoint}{RESET}")
        skip_stage1 = True
    else:
        skip_stage1 = False
    
    total_start = time.time()
    
    # Get project root and use venv Python if available
    project_root = Path(__file__).parent.absolute()
    venv_python = project_root / "venv" / "Scripts" / "python.exe"
    python_executable = str(venv_python) if venv_python.exists() else sys.executable
    
    # Stage 1: Pretraining
    if not skip_stage1:
        pretrain_cmd = [
            python_executable,
            'training_scripts/pretrain_dqn_imitation.py',
            '--size', str(args.size),
            '--mines', str(args.mines),
            '--expert-episodes', str(args.expert_episodes),
            '--num-updates', str(args.num_updates)
        ]
        
        success = run_command(pretrain_cmd, "STAGE 1: Pretraining (Imitation Learning)")
        
        if not success:
            print(f"\n{RED}Pipeline failed at Stage 1 (Pretraining){RESET}")
            sys.exit(1)
    
    # Verify checkpoint exists
    if not Path(pretrained_checkpoint).exists():
        print(f"\n{RED}Error: Pretrained checkpoint not found: {pretrained_checkpoint}{RESET}")
        print(f"{RED}Stage 1 may have failed silently.{RESET}")
        sys.exit(1)
    
    # Stage 2: RL Training
    train_cmd = [
        python_executable,
        'training_scripts/train_dqn.py',
        '--size', str(args.size),
        '--mines', str(args.mines),
        '--num-steps', str(args.rl_steps),
        '--num-envs', str(args.num_envs),
        '--checkpoint', pretrained_checkpoint
    ]
    
    success = run_command(train_cmd, "STAGE 2: RL Training (Deep Q-Learning)")
    
    if not success:
        print(f"\n{RED}Pipeline failed at Stage 2 (RL Training){RESET}")
        sys.exit(1)
    
    # Success!
    total_time = time.time() - total_start
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print_header("PIPELINE COMPLETE")
    print(f"{GREEN}[SUCCESS] Both stages completed successfully!{RESET}")
    print(f"\nTotal time: {hours}h {minutes}m {seconds}s")
    print(f"\n{CYAN}Next steps:{RESET}")
    print(f"  1. Evaluate pretrained model:")
    print(f"     python evaluate_dqn.py --size {args.size} --mines {args.mines} --episodes 1000 --checkpoint {pretrained_checkpoint}")
    print(f"\n  2. Evaluate final trained model:")
    print(f"     python evaluate_dqn.py --size {args.size} --mines {args.mines} --episodes 1000 --checkpoint dqn-checkpoint-s{args.size}-m{args.mines}-{args.rl_steps}.pth")
    print(f"\n  3. Compare with hybrid mode enabled (optional):")
    print(f"     # Edit evaluate_dqn.py: set agent.use_hybrid=True, agent.safe_action_prob=0.3")
    print(f"     python evaluate_dqn.py --size {args.size} --mines {args.mines} --episodes 1000 --checkpoint dqn-checkpoint-s{args.size}-m{args.mines}-{args.rl_steps}.pth\n")
    
    sys.exit(0)


if __name__ == "__main__":
    main()
