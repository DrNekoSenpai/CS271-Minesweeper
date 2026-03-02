"""
Train Supervised Mine Predictor

Learns to classify tiles as safe/mine based on board patterns.
Uses ground truth mine locations from completed games.

Usage:
    python training_scripts/train_supervised.py --size 5 --mines 5 --epochs 50
"""

import argparse
import os
import sys
import time
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Tuple, Dict
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.minesweeper import Minesweeper
from agents.supervised_prediction_agent import SupervisedMinePredictor
from agents.bayesian_approximation_agent import Agent as BayesianAgent


class MinesweeperDataset(Dataset):
    """
    Dataset of Minesweeper board states with ground truth mine labels
    
    Each sample: (board_state, mine_labels, legal_mask)
    """
    
    def __init__(self, data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
        """
        Args:
            data: List of (obs, mines, legal) tuples
                obs: (H, W, D) board observation
                mines: (H, W, D) binary mine map
                legal: (H, W, D) binary legal action mask
        """
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        obs, mines, legal = self.data[idx]
        return obs.copy(), mines.copy(), legal.copy()


def generate_training_data(
    size: int,
    num_mines: int,
    num_games: int,
    agent_type: str = 'bayesian',
    max_steps_per_game: int = 200,
    seed: int = None,
    verbose: bool = True
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generate training data by playing games and recording states
    
    Args:
        size: Board dimension (size x size x size)
        num_mines: Number of mines
        num_games: Number of games to play
        agent_type: 'bayesian' or 'random'
        max_steps_per_game: Max moves per game
        seed: Random seed
        verbose: Print progress
        
    Returns:
        List of (observation, mine_map, legal_mask) tuples
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Initialize agent
    if agent_type == 'bayesian':
        agent = BayesianAgent(None, size, size, size, num_mines)
    else:
        agent = None  # Will use random
    
    data = []
    games_completed = 0
    games_won = 0
    total_samples = 0
    
    print(f"\nGenerating training data from {num_games} games...")
    print(f"Agent: {agent_type}, Board: {size}x{size}x{size}, Mines: {num_mines}")
    
    start_time = time.time()
    
    for game_idx in tqdm(range(num_games), desc="Generating games", unit="game"):
        game = Minesweeper(size, size, size, num_mines)
        game_states = []
        
        for step in range(max_steps_per_game):
            # Record current state
            obs = game.get_observation()
            legal_mask = np.zeros((size, size, size), dtype=np.float32)
            
            # Mark legal actions
            for x in range(size):
                for y in range(size):
                    for z in range(size):
                        if obs[x, y, z] == -1:  # Unrevealed
                            legal_mask[x, y, z] = 1.0
            
            # Only save if there are legal actions
            if legal_mask.sum() > 0:
                game_states.append((obs.copy(), legal_mask.copy()))
            
            # Select action
            if agent is not None:
                action = agent.select_action(obs)
            else:
                # Random legal action
                legal_indices = np.argwhere(obs == -1)
                if len(legal_indices) == 0:
                    break
                idx = np.random.randint(len(legal_indices))
                x, y, z = legal_indices[idx]
                action = z * (size * size) + y * size + x
            
            # Execute action
            z = action // (size * size)
            y = (action % (size * size)) // size
            x = action % size
            
            game.reveal(x, y, z)
            done = game.game_over
            
            if done:
                games_completed += 1
                if game.win:
                    games_won += 1
                break
        
        # Now we know all mine locations - create ground truth labels from board
        mine_map = np.zeros((size, size, size), dtype=np.float32)
        mine_map[game.board == -1] = 1.0
        
        # Add all states from this game to dataset
        for obs, legal_mask in game_states:
            data.append((obs, mine_map, legal_mask))
            total_samples += 1

    
    elapsed = time.time() - start_time
    win_rate = 100.0 * games_won / max(1, games_completed)
    
    print(f"\nData generation complete!")
    print(f"  Total samples: {total_samples}")
    print(f"  Games completed: {games_completed}")
    print(f"  Win rate: {win_rate:.1f}%")
    print(f"  Time: {elapsed:.1f}s ({games_completed/elapsed:.1f} games/s)")
    
    return data


def evaluate_agent(
    agent: SupervisedMinePredictor,
    size: int,
    num_mines: int,
    num_games: int = 100,
    max_steps: int = 200
) -> Dict:
    """
    Evaluate agent by playing games
    
    Returns:
        Dict with detailed game performance metrics
    """
    wins = 0
    total_tiles = 0
    total_steps = 0
    tiles_revealed_list = []
    
    for _ in tqdm(range(num_games), desc="Evaluating", leave=False):
        game = Minesweeper(size, size, size, num_mines)
        
        for step in range(max_steps):
            obs = game.get_observation()
            action = agent.select_action(obs)
            
            z = action // (size * size)
            y = (action % (size * size)) // size
            x = action % size
            
            game.reveal(x, y, z)
            done = game.game_over
            
            if done:
                if game.win:
                    wins += 1
                revealed = np.sum(game.visible >= 0)
                total_tiles += revealed
                total_steps += step + 1
                tiles_revealed_list.append(revealed)
                break
    
    tiles_array = np.array(tiles_revealed_list) if tiles_revealed_list else np.array([0])
    
    return {
        'win_rate': 100.0 * wins / num_games,
        'num_wins': wins,
        'num_games': num_games,
        'avg_tiles_revealed': total_tiles / num_games,
        'std_tiles_revealed': np.std(tiles_array),
        'avg_game_length': total_steps / num_games,
        'best_tiles': np.max(tiles_array) if len(tiles_array) > 0 else 0,
        'worst_tiles': np.min(tiles_array) if len(tiles_array) > 0 else 0
    }


def train_epoch(
    agent: SupervisedMinePredictor,
    dataloader: DataLoader,
    device: torch.device
) -> Dict:
    """Train for one epoch"""
    agent.net.train()
    
    epoch_loss = 0.0
    epoch_acc = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        boards, mine_labels, legal_masks = batch
        boards = boards.numpy()
        mine_labels = mine_labels.numpy()
        legal_masks = legal_masks.numpy()
        
        # Train step
        stats = agent.train_batch(boards, mine_labels, legal_masks)
        
        epoch_loss += stats['loss']
        epoch_acc += stats['accuracy']
        num_batches += 1
    
    return {
        'loss': epoch_loss / num_batches,
        'accuracy': epoch_acc / num_batches
    }


def validate(
    agent: SupervisedMinePredictor,
    dataloader: DataLoader,
    device: torch.device,
    compute_classification_metrics: bool = False
) -> Dict:
    """Validate on validation set with optional detailed classification metrics"""
    agent.net.eval()
    
    val_loss = 0.0
    val_acc = 0.0
    num_batches = 0
    
    # For classification metrics
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            boards, mine_labels, legal_masks = batch
            boards_np = boards.numpy()
            mine_labels_np = mine_labels.numpy()
            legal_masks_np = legal_masks.numpy()
            
            # Convert to tensors
            boards_t = agent._obs_to_tensor(boards_np)
            B = boards_np.shape[0]
            mine_labels_flat = mine_labels_np.reshape(B, -1)
            legal_masks_flat = legal_masks_np.reshape(B, -1)
            
            targets = 1.0 - mine_labels_flat
            targets_t = torch.FloatTensor(targets).to(device)
            legal_t = torch.BoolTensor(legal_masks_flat).to(device)
            
            # Forward pass
            probs = agent.net(boards_t)
            
            # Compute metrics
            loss = F.binary_cross_entropy(probs[legal_t], targets_t[legal_t])
            preds = (probs > 0.5).float()
            acc = (preds[legal_t] == targets_t[legal_t]).float().mean()
            
            val_loss += loss.item()
            val_acc += acc.item()
            num_batches += 1
            
            # Collect for classification metrics
            if compute_classification_metrics:
                all_preds.append(preds[legal_t].cpu().numpy())
                all_targets.append(targets_t[legal_t].cpu().numpy())
    
    results = {
        'loss': val_loss / num_batches,
        'accuracy': val_acc / num_batches
    }
    
    # Compute detailed classification metrics
    if compute_classification_metrics and len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        # Safe tile metrics (target = 1)
        safe_preds = (all_preds == 1)
        safe_targets = (all_targets == 1)
        tp_safe = np.sum(safe_preds & safe_targets)
        fp_safe = np.sum(safe_preds & ~safe_targets)
        fn_safe = np.sum(~safe_preds & safe_targets)
        
        safe_precision = tp_safe / (tp_safe + fp_safe) if (tp_safe + fp_safe) > 0 else 0.0
        safe_recall = tp_safe / (tp_safe + fn_safe) if (tp_safe + fn_safe) > 0 else 0.0
        safe_f1 = 2 * safe_precision * safe_recall / (safe_precision + safe_recall) if (safe_precision + safe_recall) > 0 else 0.0
        
        # Mine tile metrics (target = 0)
        mine_preds = (all_preds == 0)
        mine_targets = (all_targets == 0)
        tp_mine = np.sum(mine_preds & mine_targets)
        fp_mine = np.sum(mine_preds & ~mine_targets)
        fn_mine = np.sum(~mine_preds & mine_targets)
        
        mine_precision = tp_mine / (tp_mine + fp_mine) if (tp_mine + fp_mine) > 0 else 0.0
        mine_recall = tp_mine / (tp_mine + fn_mine) if (tp_mine + fn_mine) > 0 else 0.0
        mine_f1 = 2 * mine_precision * mine_recall / (mine_precision + mine_recall) if (mine_precision + mine_recall) > 0 else 0.0
        
        results.update({
            'safe_precision': safe_precision,
            'safe_recall': safe_recall,
            'safe_f1': safe_f1,
            'mine_precision': mine_precision,
            'mine_recall': mine_recall,
            'mine_f1': mine_f1
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train Supervised Mine Predictor')
    
    # Environment
    parser.add_argument('--size', type=int, default=5, help='Board size (default: 5)')
    parser.add_argument('--mines', type=int, default=5, help='Number of mines (default: 5)')
    
    # Data generation
    parser.add_argument('--num-games', type=int, default=10000, 
                        help='Number of games to generate (default: 10000)')
    parser.add_argument('--agent-type', type=str, default='bayesian', 
                        choices=['bayesian', 'random'],
                        help='Agent to generate data (default: bayesian)')
    parser.add_argument('--max-steps-per-game', type=int, default=200,
                        help='Max steps per game (default: 200)')
    parser.add_argument('--data-seed', type=int, default=None,
                        help='Random seed for data generation (default: None)')
    parser.add_argument('--load-data', type=str, default=None,
                        help='Load pre-generated data from file (default: None)')
    parser.add_argument('--save-data', type=str, default=None,
                        help='Save generated data to file (default: None)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Train/val split ratio (default: 0.8)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers (default: 4)')
    parser.add_argument('--train-seed', type=int, default=42,
                        help='Random seed for training (default: 42)')
    
    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_supervised',
                        help='Directory to save checkpoints (default: checkpoints_supervised)')
    parser.add_argument('--checkpoint-freq', type=int, default=5,
                        help='Save checkpoint every N epochs (default: 5)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path (default: None)')
    
    # Evaluation
    parser.add_argument('--eval-freq', type=int, default=10,
                        help='Evaluate by playing games every N epochs (default: 10)')
    parser.add_argument('--eval-games', type=int, default=100,
                        help='Number of games for evaluation (default: 100)')
    parser.add_argument('--early-stop-patience', type=int, default=10,
                        help='Early stopping patience in epochs (default: 10)')
    
    # Metrics
    parser.add_argument('--no-metrics', action='store_true',
                        help='Disable metrics logging')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu, default: auto)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    
    args = parser.parse_args()
    
    # Setup device
    if args.no_cuda:
        device = torch.device('cpu')
    elif args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print("SUPERVISED MINE PREDICTOR TRAINING")
    print("=" * 80)
    print(f"Board: {args.size}x{args.size}x{args.size}, Mines: {args.mines}")
    print(f"Device: {device}")
    print()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create metrics directory (matches DQN format)
    metrics_enabled = not args.no_metrics
    if metrics_enabled:
        metrics_subdir = f'./metrics/s{args.size}-m{args.mines}'
        os.makedirs(metrics_subdir, exist_ok=True)
        
        # Initialize CSV files
        training_metrics_file = os.path.join(metrics_subdir, 'training_metrics.csv')
        evaluation_metrics_file = os.path.join(metrics_subdir, 'evaluation_metrics.csv')
        classification_metrics_file = os.path.join(metrics_subdir, 'classification_metrics.csv')
        
        # Write headers
        with open(training_metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 
                           'epoch_time', 'cumulative_time', 'samples_seen', 'learning_rate'])
        
        with open(evaluation_metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'win_rate', 'num_wins', 'num_games', 'avg_tiles_revealed', 
                           'std_tiles_revealed', 'avg_game_length', 'best_tiles', 'worst_tiles'])
        
        with open(classification_metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'safe_precision', 'safe_recall', 'safe_f1', 
                           'mine_precision', 'mine_recall', 'mine_f1'])
        
        print(f"\nMetrics will be saved to: {metrics_subdir}")
        cumulative_time = 0.0
    
    # Generate or load data
    if args.load_data and os.path.exists(args.load_data):
        print(f"Loading data from {args.load_data}...")
        data = torch.load(args.load_data)
        print(f"Loaded {len(data)} samples")
    else:
        data = generate_training_data(
            size=args.size,
            num_mines=args.mines,
            num_games=args.num_games,
            agent_type=args.agent_type,
            max_steps_per_game=args.max_steps_per_game,
            seed=args.data_seed,
            verbose=True
        )
        
        if args.save_data:
            print(f"\nSaving data to {args.save_data}...")
            torch.save(data, args.save_data)
    
    # Create dataset and split
    dataset = MinesweeperDataset(data)
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    torch.manual_seed(args.train_seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    # Initialize agent
    agent = SupervisedMinePredictor(
        height=args.size,
        width=args.size,
        depth=args.size,
        lr=args.lr,
        device=device
    )
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    best_checkpoint_path = None
    most_recent_checkpoint_path = None
    patience_counter = 0
    
    # Auto-detect latest checkpoint if resume is specified
    resume_path = args.resume
    if args.resume:
        if os.path.exists(args.resume):
            resume_path = args.resume
        else:
            # Try to find latest checkpoint in checkpoint dir
            checkpoint_pattern = f'supervised-s{args.size}-m{args.mines}-epoch'
            existing_checkpoints = sorted(
                [f for f in os.listdir(args.checkpoint_dir) 
                 if f.startswith(checkpoint_pattern) and f.endswith('.pth')],
                key=lambda x: int(x.split('epoch')[1].split('.')[0])
            )
            if existing_checkpoints:
                resume_path = os.path.join(args.checkpoint_dir, existing_checkpoints[-1])
                print(f"Auto-detected checkpoint: {resume_path}")
            else:
                print(f"Warning: --resume specified but no checkpoint found matching {checkpoint_pattern}")
                resume_path = None
    
    if resume_path and os.path.exists(resume_path):
        print(f"\nResuming from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)
        agent.net.load_state_dict(checkpoint['net_state'])
        agent.optim.load_state_dict(checkpoint['optim_state'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Try to reconstruct checkpoint paths
        if 'best_checkpoint_path' in checkpoint:
            best_checkpoint_path = checkpoint['best_checkpoint_path']
        if 'most_recent_checkpoint_path' in checkpoint:
            most_recent_checkpoint_path = checkpoint['most_recent_checkpoint_path']
        
        print(f"Resuming from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")
    
    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Train
        train_stats = train_epoch(agent, train_loader, device)
        
        # Validate (with classification metrics every 5 epochs)
        compute_class_metrics = (epoch + 1) % 5 == 0
        val_stats = validate(agent, val_loader, device, compute_classification_metrics=compute_class_metrics)
        
        epoch_time = time.time() - epoch_start
        if metrics_enabled:
            cumulative_time += epoch_time
        
        # Calculate samples seen
        samples_seen = (epoch + 1) * len(train_dataset)
        
        # Get current learning rate
        current_lr = agent.optim.param_groups[0]['lr']
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s, total: {cumulative_time/60:.1f}m)")
        print(f"  Train - Loss: {train_stats['loss']:.4f}, Acc: {train_stats['accuracy']*100:.2f}%")
        print(f"  Val   - Loss: {val_stats['loss']:.4f}, Acc: {val_stats['accuracy']*100:.2f}%")
        
        # Log training metrics
        if metrics_enabled:
            with open(training_metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, train_stats['loss'], train_stats['accuracy'],
                               val_stats['loss'], val_stats['accuracy'], epoch_time,
                               cumulative_time, samples_seen, current_lr])
        
        # Log classification metrics
        if compute_class_metrics and 'safe_precision' in val_stats:
            print(f"  Classification Metrics:")
            print(f"    Safe  - P: {val_stats['safe_precision']*100:.1f}%, R: {val_stats['safe_recall']*100:.1f}%, F1: {val_stats['safe_f1']:.3f}")
            print(f"    Mine  - P: {val_stats['mine_precision']*100:.1f}%, R: {val_stats['mine_recall']*100:.1f}%, F1: {val_stats['mine_f1']:.3f}")
            
            if metrics_enabled:
                with open(classification_metrics_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1, val_stats['safe_precision'], val_stats['safe_recall'],
                                   val_stats['safe_f1'], val_stats['mine_precision'], 
                                   val_stats['mine_recall'], val_stats['mine_f1']])
        
        # Save checkpoint
        is_best = val_stats['loss'] < best_val_loss
        
        # Always save current checkpoint as most recent
        checkpoint_path = os.path.join(args.checkpoint_dir, 
                                      f'supervised-s{args.size}-m{args.mines}-epoch{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'net_state': agent.net.state_dict(),
            'optim_state': agent.optim.state_dict(),
            'train_stats': train_stats,
            'val_stats': val_stats,
            'best_val_loss': best_val_loss if not is_best else val_stats['loss'],
            'best_checkpoint_path': best_checkpoint_path,
            'most_recent_checkpoint_path': checkpoint_path,
            'args': vars(args)
        }, checkpoint_path)
        
        # Delete previous most recent checkpoint (not the best one)
        if most_recent_checkpoint_path and os.path.exists(most_recent_checkpoint_path):
            if most_recent_checkpoint_path != best_checkpoint_path:
                os.remove(most_recent_checkpoint_path)
                print(f"  [Cleanup] Deleted previous checkpoint")
        
        most_recent_checkpoint_path = checkpoint_path
        
        if is_best:
            # Delete previous best checkpoint if it exists
            if best_checkpoint_path and os.path.exists(best_checkpoint_path):
                if best_checkpoint_path != most_recent_checkpoint_path:
                    os.remove(best_checkpoint_path)
                    print(f"  [Cleanup] Deleted previous best checkpoint")
            
            best_val_loss = val_stats['loss']
            best_checkpoint_path = checkpoint_path
            patience_counter = 0
            print(f"  ★ New best model! (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  Checkpoint saved: epoch {epoch+1}")
        
        # Game evaluation
        if (epoch + 1) % args.eval_freq == 0:
            print(f"  Evaluating on {args.eval_games} games...")
            eval_stats = evaluate_agent(agent, args.size, args.mines, args.eval_games)
            print(f"  Game Performance:")
            print(f"    Win rate: {eval_stats['win_rate']:.2f}% ({eval_stats['num_wins']}/{args.eval_games})")
            print(f"    Avg tiles revealed: {eval_stats['avg_tiles_revealed']:.1f} ± {eval_stats['std_tiles_revealed']:.1f}")
            print(f"    Range: [{eval_stats['worst_tiles']}, {eval_stats['best_tiles']}] tiles")
            print(f"    Avg game length: {eval_stats['avg_game_length']:.1f} moves")
            
            if metrics_enabled:
                with open(evaluation_metrics_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1, eval_stats['win_rate'], eval_stats['num_wins'],
                                   eval_stats['num_games'], eval_stats['avg_tiles_revealed'],
                                   eval_stats['std_tiles_revealed'], eval_stats['avg_game_length'],
                                   eval_stats['best_tiles'], eval_stats['worst_tiles']])
        
        # Early stopping
        if patience_counter >= args.early_stop_patience:
            print(f"\n  Early stopping triggered (no improvement for {args.early_stop_patience} epochs)")
            break
        
        print()
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    
    # Load best model (it's the one with the lowest loss, which is tracked in best_checkpoint_path)
    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        print(f"Loading best model from {best_checkpoint_path}...")
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        agent.net.load_state_dict(checkpoint['net_state'])
    else:
        print("No best checkpoint found, using current model...")
    
    print(f"\nEvaluating on {args.eval_games * 5} games...")
    final_stats = evaluate_agent(agent, args.size, args.mines, args.eval_games * 5)
    
    print(f"\nFinal Performance:")
    print(f"  Win rate: {final_stats['win_rate']:.2f}% ({final_stats['num_wins']}/{args.eval_games * 5})")
    print(f"  Avg tiles revealed: {final_stats['avg_tiles_revealed']:.1f} ± {final_stats['std_tiles_revealed']:.1f}")
    print(f"  Range: [{final_stats['worst_tiles']}, {final_stats['best_tiles']}] tiles")
    print(f"  Avg game length: {final_stats['avg_game_length']:.1f} moves")
    
    if metrics_enabled:
        # Log final evaluation
        with open(evaluation_metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['FINAL', final_stats['win_rate'], final_stats['num_wins'],
                           args.eval_games * 5, final_stats['avg_tiles_revealed'],
                           final_stats['std_tiles_revealed'], final_stats['avg_game_length'],
                           final_stats['best_tiles'], final_stats['worst_tiles']])
        
        print(f"\nMetrics saved to: {metrics_subdir}")
    
    print()
    print("Training complete!")


if __name__ == '__main__':
    main()
