#!/usr/bin/env python3
"""
Script independiente para visualizar las pérdidas guardadas durante el entrenamiento.
Útil para generar gráficos actualizados sin necesidad de ejecutar el entrenamiento.
"""

import json
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np

def load_losses(loss_dir):
    """Load losses from JSON files"""
    epoch_file = os.path.join(loss_dir, 'epoch_losses.json')
    batch_file = os.path.join(loss_dir, 'batch_losses.json')
    
    epoch_losses = None
    batch_losses = None
    
    if os.path.exists(epoch_file):
        with open(epoch_file, 'r') as f:
            epoch_losses = json.load(f)
        print(f"Loaded epoch losses: {len(epoch_losses['epoch'])} epochs")
    else:
        print(f"Warning: {epoch_file} not found")
    
    if os.path.exists(batch_file):
        with open(batch_file, 'r') as f:
            batch_losses = json.load(f)
        print(f"Loaded batch losses: {len(batch_losses['iteration'])} iterations")
    else:
        print(f"Warning: {batch_file} not found")
    
    return epoch_losses, batch_losses

def plot_epoch_losses(epoch_losses, save_dir):
    """Generate epoch-level loss plots"""
    if epoch_losses is None or len(epoch_losses['epoch']) == 0:
        print("No epoch losses to plot")
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CycleGAN Training Losses (Epoch Level)', fontsize=16)
    
    epochs = epoch_losses['epoch']
    
    # Plot 1: Generator and Discriminator losses
    axes[0, 0].plot(epochs, epoch_losses['loss_G'], 'b-', label='Generator Loss', linewidth=2)
    axes[0, 0].plot(epochs, epoch_losses['loss_D'], 'r-', label='Discriminator Loss', linewidth=2)
    axes[0, 0].set_title('Generator vs Discriminator Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: GAN losses
    axes[0, 1].plot(epochs, epoch_losses['loss_GAN_A2B'], 'g-', label='GAN A2B Loss', linewidth=2)
    axes[0, 1].plot(epochs, epoch_losses['loss_GAN_B2A'], 'm-', label='GAN B2A Loss', linewidth=2)
    axes[0, 1].set_title('GAN Adversarial Losses')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Cycle consistency losses (L1)
    axes[0, 2].plot(epochs, epoch_losses['loss_cycle_A_L1'], 'c-', label='Cycle A L1 Loss', linewidth=2)
    axes[0, 2].plot(epochs, epoch_losses['loss_cycle_B_L1'], 'orange', label='Cycle B L1 Loss', linewidth=2)
    axes[0, 2].set_title('Cycle Consistency Losses (L1)')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Identity losses
    axes[1, 0].plot(epochs, epoch_losses['loss_idt_A'], 'purple', label='Identity A Loss', linewidth=2)
    axes[1, 0].plot(epochs, epoch_losses['loss_idt_B'], 'brown', label='Identity B Loss', linewidth=2)
    axes[1, 0].set_title('Identity Losses')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: VGG losses (if available)
    if 'loss_vgg_total' in epoch_losses and any(val > 0 for val in epoch_losses['loss_vgg_total']):
        axes[1, 1].plot(epochs, epoch_losses['loss_cycle_A_vgg'], 'navy', label='Cycle A VGG Loss', linewidth=2)
        axes[1, 1].plot(epochs, epoch_losses['loss_cycle_B_vgg'], 'darkred', label='Cycle B VGG Loss', linewidth=2)
        axes[1, 1].plot(epochs, epoch_losses['loss_vgg_total'], 'black', label='Total VGG Loss', linewidth=2)
        axes[1, 1].set_title('VGG Perceptual Losses')
    else:
        axes[1, 1].text(0.5, 0.5, 'VGG Loss Disabled', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('VGG Perceptual Losses (Disabled)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Learning rates
    if 'lr_G' in epoch_losses and 'lr_D' in epoch_losses:
        axes[1, 2].plot(epochs, epoch_losses['lr_G'], 'blue', label='Generator LR', linewidth=2)
        axes[1, 2].plot(epochs, epoch_losses['lr_D'], 'red', label='Discriminator LR', linewidth=2)
        axes[1, 2].set_title('Learning Rates')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Learning Rate')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_yscale('log')
    else:
        axes[1, 2].text(0.5, 0.5, 'LR Data Not Available', ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Learning Rates')
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(save_dir, 'epoch_losses_plot.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Epoch losses plot saved to: {plot_file}")

def plot_batch_losses(batch_losses, save_dir, n_recent=2000):
    """Generate batch-level loss plots"""
    if batch_losses is None or len(batch_losses['iteration']) == 0:
        print("No batch losses to plot")
        return
    
    # Take recent iterations
    n_recent = min(n_recent, len(batch_losses['iteration']))
    recent_iterations = batch_losses['iteration'][-n_recent:]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Recent Batch Losses (Last {n_recent} Iterations)', fontsize=14)
    
    # Plot 1: Generator and Discriminator losses
    axes[0, 0].plot(recent_iterations, batch_losses['loss_G'][-n_recent:], 'b-', alpha=0.7, linewidth=1)
    axes[0, 0].plot(recent_iterations, batch_losses['loss_D'][-n_recent:], 'r-', alpha=0.7, linewidth=1)
    axes[0, 0].set_title('Generator vs Discriminator Loss')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(['Generator', 'Discriminator'])
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: GAN losses
    axes[0, 1].plot(recent_iterations, batch_losses['loss_GAN_A2B'][-n_recent:], 'g-', alpha=0.7, linewidth=1)
    axes[0, 1].plot(recent_iterations, batch_losses['loss_GAN_B2A'][-n_recent:], 'm-', alpha=0.7, linewidth=1)
    axes[0, 1].set_title('GAN Adversarial Losses')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend(['GAN A2B', 'GAN B2A'])
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Cycle losses
    axes[1, 0].plot(recent_iterations, batch_losses['loss_cycle_A_L1'][-n_recent:], 'c-', alpha=0.7, linewidth=1)
    axes[1, 0].plot(recent_iterations, batch_losses['loss_cycle_B_L1'][-n_recent:], 'orange', alpha=0.7, linewidth=1)
    axes[1, 0].set_title('Cycle Consistency Losses')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend(['Cycle A L1', 'Cycle B L1'])
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Identity losses
    axes[1, 1].plot(recent_iterations, batch_losses['loss_idt_A'][-n_recent:], 'purple', alpha=0.7, linewidth=1)
    axes[1, 1].plot(recent_iterations, batch_losses['loss_idt_B'][-n_recent:], 'brown', alpha=0.7, linewidth=1)
    axes[1, 1].set_title('Identity Losses')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend(['Identity A', 'Identity B'])
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the batch plot
    batch_plot_file = os.path.join(save_dir, 'batch_losses_plot.png')
    plt.savefig(batch_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Batch losses plot saved to: {batch_plot_file}")

def plot_loss_summary(epoch_losses, save_dir):
    """Generate a summary plot with key metrics"""
    if epoch_losses is None or len(epoch_losses['epoch']) == 0:
        print("No epoch losses for summary plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('CycleGAN Training Summary', fontsize=16)
    
    epochs = epoch_losses['epoch']
    
    # Plot 1: Main losses
    axes[0, 0].plot(epochs, epoch_losses['loss_G'], 'b-', label='Generator', linewidth=2)
    axes[0, 0].plot(epochs, epoch_losses['loss_D'], 'r-', label='Discriminator', linewidth=2)
    axes[0, 0].set_title('Main Losses')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Cycle losses
    total_cycle_L1 = [a + b for a, b in zip(epoch_losses['loss_cycle_A_L1'], epoch_losses['loss_cycle_B_L1'])]
    axes[0, 1].plot(epochs, total_cycle_L1, 'c-', label='Total Cycle L1', linewidth=2)
    if 'loss_vgg_total' in epoch_losses:
        axes[0, 1].plot(epochs, epoch_losses['loss_vgg_total'], 'orange', label='Total VGG', linewidth=2)
    axes[0, 1].set_title('Cycle Consistency Losses')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Training time per epoch
    if 'epoch_duration' in epoch_losses:
        axes[1, 0].plot(epochs, epoch_losses['epoch_duration'], 'g-', linewidth=2)
        axes[1, 0].set_title('Training Time per Epoch')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Duration Data Not Available', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Training Time per Epoch')
    
    # Plot 4: Loss ratio (G/D)
    loss_ratio = [g/d if d > 0 else 0 for g, d in zip(epoch_losses['loss_G'], epoch_losses['loss_D'])]
    axes[1, 1].plot(epochs, loss_ratio, 'purple', linewidth=2)
    axes[1, 1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Balanced (G=D)')
    axes[1, 1].set_title('Generator/Discriminator Loss Ratio')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('G Loss / D Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the summary plot
    summary_plot_file = os.path.join(save_dir, 'training_summary.png')
    plt.savefig(summary_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training summary plot saved to: {summary_plot_file}")

def main():
    parser = argparse.ArgumentParser(description='Plot CycleGAN training losses from saved data')
    parser.add_argument('--loss_dir', type=str, default='outputs/loss_tracking',
                        help='Directory containing loss JSON files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save plots (default: same as loss_dir)')
    parser.add_argument('--batch_recent', type=int, default=2000,
                        help='Number of recent batch iterations to plot')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.loss_dir):
        print(f"Error: Loss directory {args.loss_dir} does not exist")
        return
    
    output_dir = args.output_dir if args.output_dir else args.loss_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading losses from: {args.loss_dir}")
    print(f"Saving plots to: {output_dir}")
    
    # Load losses
    epoch_losses, batch_losses = load_losses(args.loss_dir)
    
    # Generate plots
    if epoch_losses:
        plot_epoch_losses(epoch_losses, output_dir)
        plot_loss_summary(epoch_losses, output_dir)
    
    if batch_losses:
        plot_batch_losses(batch_losses, output_dir, args.batch_recent)
    
    print("Plotting completed!")

if __name__ == "__main__":
    main() 