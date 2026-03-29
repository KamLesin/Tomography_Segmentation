"""
Training script for liver segmentation model.
"""
import argparse
import os
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd

from model import UNet
from dataset import get_data_loaders


def calculate_class_weights(metadata_csv, data_root, patient_ids):
    """
    Calculate class weights based on pixel frequency in training data.
    Inverse frequency weighting: weight = 1 / frequency
    
    Returns:
        torch.Tensor of shape (n_classes,) with weights
    """
    from dataset import LiverDataset
    
    # Create temporary dataset to count pixels
    dataset = LiverDataset(
        metadata_csv=metadata_csv,
        data_root=data_root,
        patient_ids=patient_ids,
        img_size=(512, 512),
        normalize=True,
        multiclass=True,
    )
    
    class_counts = np.array([0, 0, 0], dtype=np.float64)  # 3 classes
    
    print("Calculating class weights from training data...")
    for i in range(len(dataset)):
        batch = dataset[i]
        label = batch['label'].numpy()
        
        for c in range(3):
            class_counts[c] += (label == c).sum()
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(dataset)} samples")
    
    # Calculate weights (inverse frequency)
    total_pixels = class_counts.sum()
    weights = total_pixels / (class_counts * 3)  # Normalize
    weights = torch.from_numpy(weights).float()
    
    print(f"\nClass distribution:")
    for c in range(3):
        pct = 100 * class_counts[c] / total_pixels
        print(f"  Class {c}: {class_counts[c]:.0f} pixels ({pct:.2f}%), weight={weights[c]:.4f}")
    
    return weights


def dice_coefficient(pred, target, smooth=1e-5):
    """
    Calculate Dice coefficient for binary masks.
    
    Args:
        pred: Predicted mask (B, C, H, W) - probabilities or logits
        target: Ground truth (B, H, W) - class indices
        smooth: Smoothing factor
    
    Returns:
        Dice coefficient per class
    """
    num_classes = pred.shape[1]
    dice_scores = []
    
    pred = torch.softmax(pred, dim=1)
    
    for c in range(num_classes):
        pred_c = pred[:, c]
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.item())
    
    return dice_scores


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    dice_scores = [[] for _ in range(3)]  # 3 classes
    
    for batch_idx, batch in enumerate(loader):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        dice = dice_coefficient(outputs, labels)
        for i, d in enumerate(dice):
            dice_scores[i].append(d)
        
        # Log progress
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(loader)
    avg_dice = [np.mean(scores) for scores in dice_scores]
    
    return avg_loss, avg_dice


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    
    total_loss = 0
    dice_scores = [[] for _ in range(3)]
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            dice = dice_coefficient(outputs, labels)
            for i, d in enumerate(dice):
                dice_scores[i].append(d)
    
    avg_loss = total_loss / len(loader)
    avg_dice = [np.mean(scores) for scores in dice_scores]
    
    return avg_loss, avg_dice


def main():
    parser = argparse.ArgumentParser(description='Train liver segmentation model')
    
    # Data parameters
    parser.add_argument('--metadata', type=str, required=True,
                        help='Path to metadata CSV')
    parser.add_argument('--folds-csv', type=str, required=True,
                        help='Path to CV folds CSV')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory with images and labels')
    parser.add_argument('--fold', type=int, required=True,
                        help='Fold number to use for validation (0-5)')
    
    # Model parameters
    parser.add_argument('--n-channels', type=int, default=1,
                        help='Number of input channels')
    parser.add_argument('--n-classes', type=int, default=3,
                        help='Number of output classes')
    parser.add_argument('--use-batchnorm', action='store_true',
                        help='Use batch normalization')
    parser.add_argument('--bilinear', action='store_true',
                        help='Use bilinear upsampling')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weighted-loss', action='store_true',
                        help='Use weighted loss for class imbalance')
    parser.add_argument('--img-size', type=int, default=512,
                        help='Image size (square)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    
    # Other parameters
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save-dir', type=str, default='runs',
                        help='Directory to save checkpoints and logs')
    parser.add_argument('--experiment', type=str, default='experiment',
                        help='Experiment name')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    save_path = Path(args.save_dir) / args.experiment / f"fold_{args.fold}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(save_path))
    
    # Load folds for weight calculation
    folds_df = pd.read_csv(args.folds_csv)
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader = get_data_loaders(
        metadata_csv=args.metadata,
        folds_csv=args.folds_csv,
        data_root=args.data_root,
        fold=args.fold,
        batch_size=args.batch_size,
        img_size=(args.img_size, args.img_size),
        normalize=True,
        multiclass=True,
        num_workers=args.num_workers,
    )
    
    # Create model
    print("\nCreating model...")
    model = UNet(
        n_channels=args.n_channels,
        n_classes=args.n_classes,
        bilinear=args.bilinear,
        use_batchnorm=args.use_batchnorm,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    if args.weighted_loss:
        # Calculate class weights from training data
        train_patients = folds_df[folds_df['fold'] != args.fold]['patient_id'].tolist()
        class_weights = calculate_class_weights(
            metadata_csv=args.metadata,
            data_root=args.data_root,
            patient_ids=train_patients,
        )
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"\nUsing weighted loss with weights: {class_weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("\nUsing standard (unweighted) CrossEntropyLoss")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("\nStarting training...")
    best_val_dice = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        start_time = time.time()
        
        # Train
        train_loss, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        # Log results
        print(f"Epoch {epoch+1} completed in {epoch_time:.1f}s")
        print(f"Train Loss: {train_loss:.4f}, Dice: {train_dice}")
        print(f"Val Loss: {val_loss:.4f}, Dice: {val_dice}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Dice/train_background', train_dice[0], epoch)
        writer.add_scalar('Dice/train_liver', train_dice[1], epoch)
        writer.add_scalar('Dice/train_tumor', train_dice[2], epoch)
        writer.add_scalar('Dice/val_background', val_dice[0], epoch)
        writer.add_scalar('Dice/val_liver', val_dice[1], epoch)
        writer.add_scalar('Dice/val_tumor', val_dice[2], epoch)
        
        # Save checkpoint
        checkpoint_path = save_path / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_dice': train_dice,
            'val_dice': val_dice,
        }, checkpoint_path)
        
        # Save best model
        val_dice_mean = np.mean(val_dice[1:])  # Average of liver and tumor
        if val_dice_mean > best_val_dice:
            best_val_dice = val_dice_mean
            best_path = save_path / 'best_model.pth'
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model with val dice: {best_val_dice:.4f}")
    
    writer.close()
    print("\nTraining completed!")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"Results saved to: {save_path}")


if __name__ == '__main__':
    main()
