"""
Aggregate k-fold cross validation results.
Reads checkpoint files from all folds and computes mean/std metrics.
"""
import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd


def load_fold_results(fold_dir):
    """Load best results from a fold directory."""
    # Try to load the last checkpoint
    checkpoints = list(fold_dir.glob('checkpoint_epoch_*.pth'))
    if not checkpoints:
        print(f"Warning: No checkpoints found in {fold_dir}")
        return None
    
    # Get the latest checkpoint
    latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
    
    try:
        # PyTorch 2.6 requires weights_only=False for backward compatibility
        checkpoint = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)
        return {
            'epoch': checkpoint['epoch'],
            'train_loss': checkpoint['train_loss'],
            'val_loss': checkpoint['val_loss'],
            'train_dice': checkpoint['train_dice'],
            'val_dice': checkpoint['val_dice'],
        }
    except Exception as e:
        print(f"Error loading {latest_checkpoint}: {e}")
        return None


def aggregate_results(experiment_dir, num_folds=6):
    """Aggregate results from all folds."""
    experiment_path = Path(experiment_dir)
    
    if not experiment_path.exists():
        print(f"Error: Experiment directory not found: {experiment_path}")
        return
    
    # Collect results from all folds
    results = []
    for fold_idx in range(num_folds):
        fold_dir = experiment_path / f"fold_{fold_idx}"
        if not fold_dir.exists():
            print(f"Warning: Fold {fold_idx} directory not found")
            continue
        
        fold_results = load_fold_results(fold_dir)
        if fold_results is not None:
            fold_results['fold'] = fold_idx
            results.append(fold_results)
    
    if not results:
        print("Error: No results found in any fold")
        return
    
    print(f"\nLoaded results from {len(results)}/{num_folds} folds\n")
    
    # Create DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Print individual fold results
    print("=" * 80)
    print("INDIVIDUAL FOLD RESULTS")
    print("=" * 80)
    print(f"{'Fold':<6} {'Train Loss':<12} {'Val Loss':<12} {'Train Dice':<40} {'Val Dice':<40}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        fold = int(row['fold'])
        train_loss = row['train_loss']
        val_loss = row['val_loss']
        train_dice = row['train_dice']
        val_dice = row['val_dice']
        
        train_dice_str = f"[{train_dice[0]:.3f}, {train_dice[1]:.3f}, {train_dice[2]:.3f}]"
        val_dice_str = f"[{val_dice[0]:.3f}, {val_dice[1]:.3f}, {val_dice[2]:.3f}]"
        
        print(f"{fold:<6} {train_loss:<12.4f} {val_loss:<12.4f} {train_dice_str:<40} {val_dice_str:<40}")
    
    print()
    
    # Compute aggregated statistics
    print("=" * 80)
    print("AGGREGATED RESULTS (Mean ± Std)")
    print("=" * 80)
    
    # Extract dice scores for each class
    train_dice_bg = [r['train_dice'][0] for r in results]
    train_dice_liver = [r['train_dice'][1] for r in results]
    train_dice_tumor = [r['train_dice'][2] for r in results]
    
    val_dice_bg = [r['val_dice'][0] for r in results]
    val_dice_liver = [r['val_dice'][1] for r in results]
    val_dice_tumor = [r['val_dice'][2] for r in results]
    
    # Loss
    train_loss_mean = df['train_loss'].mean()
    train_loss_std = df['train_loss'].std()
    val_loss_mean = df['val_loss'].mean()
    val_loss_std = df['val_loss'].std()
    
    print(f"\nLoss:")
    print(f"  Train: {train_loss_mean:.4f} ± {train_loss_std:.4f}")
    print(f"  Val:   {val_loss_mean:.4f} ± {val_loss_std:.4f}")
    
    # Dice scores
    print(f"\nDice - Background:")
    print(f"  Train: {np.mean(train_dice_bg):.4f} ± {np.std(train_dice_bg):.4f}")
    print(f"  Val:   {np.mean(val_dice_bg):.4f} ± {np.std(val_dice_bg):.4f}")
    
    print(f"\nDice - Liver:")
    print(f"  Train: {np.mean(train_dice_liver):.4f} ± {np.std(train_dice_liver):.4f}")
    print(f"  Val:   {np.mean(val_dice_liver):.4f} ± {np.std(val_dice_liver):.4f}")
    
    print(f"\nDice - Tumor:")
    print(f"  Train: {np.mean(train_dice_tumor):.4f} ± {np.std(train_dice_tumor):.4f}")
    print(f"  Val:   {np.mean(val_dice_tumor):.4f} ± {np.std(val_dice_tumor):.4f}")
    
    # Average of liver + tumor (often used as main metric)
    train_dice_avg = [(r['train_dice'][1] + r['train_dice'][2]) / 2 for r in results]
    val_dice_avg = [(r['val_dice'][1] + r['val_dice'][2]) / 2 for r in results]
    
    print(f"\nDice - Average (Liver + Tumor):")
    print(f"  Train: {np.mean(train_dice_avg):.4f} ± {np.std(train_dice_avg):.4f}")
    print(f"  Val:   {np.mean(val_dice_avg):.4f} ± {np.std(val_dice_avg):.4f}")
    
    print("=" * 80)
    
    # Save summary to CSV
    summary_path = experiment_path / "cv_summary.csv"
    df_summary = pd.DataFrame({
        'metric': [
            'train_loss', 'val_loss',
            'train_dice_bg', 'val_dice_bg',
            'train_dice_liver', 'val_dice_liver',
            'train_dice_tumor', 'val_dice_tumor',
            'train_dice_avg', 'val_dice_avg'
        ],
        'mean': [
            train_loss_mean, val_loss_mean,
            np.mean(train_dice_bg), np.mean(val_dice_bg),
            np.mean(train_dice_liver), np.mean(val_dice_liver),
            np.mean(train_dice_tumor), np.mean(val_dice_tumor),
            np.mean(train_dice_avg), np.mean(val_dice_avg)
        ],
        'std': [
            train_loss_std, val_loss_std,
            np.std(train_dice_bg), np.std(val_dice_bg),
            np.std(train_dice_liver), np.std(val_dice_liver),
            np.std(train_dice_tumor), np.std(val_dice_tumor),
            np.std(train_dice_avg), np.std(val_dice_avg)
        ]
    })
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate k-fold CV results')
    parser.add_argument('--experiment-dir', type=str, required=True,
                        help='Path to experiment directory (e.g., runs/baseline)')
    parser.add_argument('--num-folds', type=int, default=6,
                        help='Number of folds')
    
    args = parser.parse_args()
    aggregate_results(args.experiment_dir, args.num_folds)


if __name__ == '__main__':
    main()
