# src/train.py (con AMP, VGG Loss y Gradient Clipping habilitados)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# --- AMP Imports ---
from torch.cuda.amp import GradScaler, autocast
# -------------------

import os
import time
import itertools
from tqdm import tqdm
import argparse # For command-line arguments like local_rank
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments

# Import project modules
from src import config
from src.dataset import NpzImageDataset
from src.models import networks
# --- VGG Loss Import ---
from src.vgg_loss import VGG19PerceptualLoss, vgg_perceptual_criterion
# ----------------------
from src.utils import (
    ImagePool, get_scheduler, init_weights, save_checkpoint, load_checkpoint,
    save_sample_images, setup_ddp, cleanup_ddp, is_main_process, tensor2im
)

class LossTracker:
    """Class to track and save training losses"""
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize loss history dictionaries
        self.epoch_losses = {
            'epoch': [],
            'loss_G': [],
            'loss_D': [],
            'loss_GAN_A2B': [],
            'loss_GAN_B2A': [],
            'loss_cycle_A_L1': [],
            'loss_cycle_B_L1': [],
            'loss_cycle_A_vgg': [],
            'loss_cycle_B_vgg': [],
            'loss_idt_A': [],
            'loss_idt_B': [],
            'loss_vgg_total': [],
            'lr_G': [],
            'lr_D': [],
            'epoch_duration': []
        }
        
        self.batch_losses = {
            'iteration': [],
            'epoch': [],
            'batch': [],
            'loss_G': [],
            'loss_D': [],
            'loss_GAN_A2B': [],
            'loss_GAN_B2A': [],
            'loss_cycle_A_L1': [],
            'loss_cycle_B_L1': [],
            'loss_cycle_A_vgg': [],
            'loss_cycle_B_vgg': [],
            'loss_idt_A': [],
            'loss_idt_B': []
        }
        
    def add_batch_loss(self, iteration, epoch, batch, losses_dict):
        """Add batch-level losses"""
        self.batch_losses['iteration'].append(iteration)
        self.batch_losses['epoch'].append(epoch)
        self.batch_losses['batch'].append(batch)
        
        for key, value in losses_dict.items():
            if key in self.batch_losses:
                self.batch_losses[key].append(float(value))
    
    def add_epoch_loss(self, epoch, losses_dict, lr_G, lr_D, duration):
        """Add epoch-level losses"""
        self.epoch_losses['epoch'].append(epoch)
        self.epoch_losses['lr_G'].append(lr_G)
        self.epoch_losses['lr_D'].append(lr_D)
        self.epoch_losses['epoch_duration'].append(duration)
        
        for key, value in losses_dict.items():
            if key in self.epoch_losses:
                self.epoch_losses[key].append(float(value))
    
    def save_losses(self):
        """Save losses to JSON files"""
        # Save epoch losses
        epoch_file = os.path.join(self.save_dir, 'epoch_losses.json')
        with open(epoch_file, 'w') as f:
            json.dump(self.epoch_losses, f, indent=2)
        
        # Save batch losses (only recent ones to avoid huge files)
        batch_file = os.path.join(self.save_dir, 'batch_losses.json')
        # Keep only last 10000 batch records to avoid huge files
        if len(self.batch_losses['iteration']) > 10000:
            for key in self.batch_losses:
                self.batch_losses[key] = self.batch_losses[key][-10000:]
        
        with open(batch_file, 'w') as f:
            json.dump(self.batch_losses, f, indent=2)
    
    def plot_losses(self):
        """Generate and save loss plots"""
        if len(self.epoch_losses['epoch']) == 0:
            return
            
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('CycleGAN Training Losses', fontsize=16)
        
        epochs = self.epoch_losses['epoch']
        
        # Plot 1: Generator and Discriminator losses
        axes[0, 0].plot(epochs, self.epoch_losses['loss_G'], 'b-', label='Generator Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.epoch_losses['loss_D'], 'r-', label='Discriminator Loss', linewidth=2)
        axes[0, 0].set_title('Generator vs Discriminator Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: GAN losses
        axes[0, 1].plot(epochs, self.epoch_losses['loss_GAN_A2B'], 'g-', label='GAN A2B Loss', linewidth=2)
        axes[0, 1].plot(epochs, self.epoch_losses['loss_GAN_B2A'], 'm-', label='GAN B2A Loss', linewidth=2)
        axes[0, 1].set_title('GAN Adversarial Losses')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Cycle consistency losses (L1)
        axes[0, 2].plot(epochs, self.epoch_losses['loss_cycle_A_L1'], 'c-', label='Cycle A L1 Loss', linewidth=2)
        axes[0, 2].plot(epochs, self.epoch_losses['loss_cycle_B_L1'], 'orange', label='Cycle B L1 Loss', linewidth=2)
        axes[0, 2].set_title('Cycle Consistency Losses (L1)')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Identity losses
        axes[1, 0].plot(epochs, self.epoch_losses['loss_idt_A'], 'purple', label='Identity A Loss', linewidth=2)
        axes[1, 0].plot(epochs, self.epoch_losses['loss_idt_B'], 'brown', label='Identity B Loss', linewidth=2)
        axes[1, 0].set_title('Identity Losses')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: VGG losses (if available)
        if any(val > 0 for val in self.epoch_losses['loss_vgg_total']):
            axes[1, 1].plot(epochs, self.epoch_losses['loss_cycle_A_vgg'], 'navy', label='Cycle A VGG Loss', linewidth=2)
            axes[1, 1].plot(epochs, self.epoch_losses['loss_cycle_B_vgg'], 'darkred', label='Cycle B VGG Loss', linewidth=2)
            axes[1, 1].plot(epochs, self.epoch_losses['loss_vgg_total'], 'black', label='Total VGG Loss', linewidth=2)
            axes[1, 1].set_title('VGG Perceptual Losses')
        else:
            axes[1, 1].text(0.5, 0.5, 'VGG Loss Disabled', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('VGG Perceptual Losses (Disabled)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Learning rates
        axes[1, 2].plot(epochs, self.epoch_losses['lr_G'], 'blue', label='Generator LR', linewidth=2)
        axes[1, 2].plot(epochs, self.epoch_losses['lr_D'], 'red', label='Discriminator LR', linewidth=2)
        axes[1, 2].set_title('Learning Rates')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Learning Rate')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_yscale('log')
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(self.save_dir, 'training_losses.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a separate plot for recent batch losses (last 1000 iterations)
        if len(self.batch_losses['iteration']) > 0:
            self._plot_recent_batch_losses()
    
    def _plot_recent_batch_losses(self):
        """Plot recent batch losses for detailed monitoring"""
        if len(self.batch_losses['iteration']) == 0:
            return
            
        # Take last 1000 iterations or all if less
        n_recent = min(1000, len(self.batch_losses['iteration']))
        recent_iterations = self.batch_losses['iteration'][-n_recent:]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Recent Batch Losses (Last {} Iterations)'.format(n_recent), fontsize=14)
        
        # Plot 1: Generator and Discriminator losses
        axes[0, 0].plot(recent_iterations, self.batch_losses['loss_G'][-n_recent:], 'b-', alpha=0.7, linewidth=1)
        axes[0, 0].plot(recent_iterations, self.batch_losses['loss_D'][-n_recent:], 'r-', alpha=0.7, linewidth=1)
        axes[0, 0].set_title('Generator vs Discriminator Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend(['Generator', 'Discriminator'])
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: GAN losses
        axes[0, 1].plot(recent_iterations, self.batch_losses['loss_GAN_A2B'][-n_recent:], 'g-', alpha=0.7, linewidth=1)
        axes[0, 1].plot(recent_iterations, self.batch_losses['loss_GAN_B2A'][-n_recent:], 'm-', alpha=0.7, linewidth=1)
        axes[0, 1].set_title('GAN Adversarial Losses')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend(['GAN A2B', 'GAN B2A'])
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Cycle losses
        axes[1, 0].plot(recent_iterations, self.batch_losses['loss_cycle_A_L1'][-n_recent:], 'c-', alpha=0.7, linewidth=1)
        axes[1, 0].plot(recent_iterations, self.batch_losses['loss_cycle_B_L1'][-n_recent:], 'orange', alpha=0.7, linewidth=1)
        axes[1, 0].set_title('Cycle Consistency Losses')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend(['Cycle A L1', 'Cycle B L1'])
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Identity losses
        axes[1, 1].plot(recent_iterations, self.batch_losses['loss_idt_A'][-n_recent:], 'purple', alpha=0.7, linewidth=1)
        axes[1, 1].plot(recent_iterations, self.batch_losses['loss_idt_B'][-n_recent:], 'brown', alpha=0.7, linewidth=1)
        axes[1, 1].set_title('Identity Losses')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend(['Identity A', 'Identity B'])
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the batch plot
        batch_plot_file = os.path.join(self.save_dir, 'recent_batch_losses.png')
        plt.savefig(batch_plot_file, dpi=300, bbox_inches='tight')
        plt.close()

def main(rank, world_size):
    """Main training function with AMP, optional VGG Loss, and Gradient Clipping."""

    # --- 1. DDP Setup ---
    print(f"Initializing DDP for Rank {rank}/{world_size}...")
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu") # Pin to specific GPU or use CPU

    # --- 2. Configuration and Paths (Accessible via config) ---
    # Check if AMP is enabled in config
    amp_enabled = getattr(config, 'AMP_ENABLED', False) # Default to False if not defined
    if amp_enabled and torch.cuda.is_available():
        print(f"[Rank {rank}] Automatic Mixed Precision (AMP) ENABLED.")
    elif amp_enabled and not torch.cuda.is_available():
        print(f"[Rank {rank}] Warning: AMP requested but CUDA not available. AMP DISABLED.")
        amp_enabled = False
    else:
        print(f"[Rank {rank}] Automatic Mixed Precision (AMP) DISABLED.")

    # Check if VGG Loss is enabled
    use_vgg_loss = getattr(config, 'USE_VGG_LOSS', False)
    lambda_vgg = getattr(config, 'LAMBDA_VGG', 0.0) if use_vgg_loss else 0.0
    if use_vgg_loss and lambda_vgg > 0:
        print(f"[Rank {rank}] VGG Perceptual Loss ENABLED (Lambda_VGG={lambda_vgg}).")
    else:
        print(f"[Rank {rank}] VGG Perceptual Loss DISABLED.")
        use_vgg_loss = False # Ensure flag is False if lambda is 0


    # --- 3. Dataset and DataLoader ---
    print(f"[Rank {rank}] Loading dataset...")
    try:
        train_dataset = NpzImageDataset(npz_path=config.TRAIN_DATA_PATH, mode='train')
    except Exception as e:
        print(f"[Rank {rank}] FATAL ERROR loading training dataset: {e}")
        cleanup_ddp()
        return # Exit process if dataset fails to load

    # Distributed Sampler ensures each GPU gets a different part of the data
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True # Shuffle data for training
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE, # Batch size PER GPU
        sampler=train_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False, # Usually improves performance by speeding up CPU->GPU transfer
        drop_last=True # Drop last incomplete batch if dataset size isn't divisible by total batch size
    )
    print(f"[Rank {rank}] DataLoader initialized. Batches per epoch: {len(train_loader)}")

    # --- 4. Model Definition ---
    print(f"[Rank {rank}] Defining models...")
    # Create models on CPU first before moving to device
    netG_A2B = networks.define_G(config.INPUT_CHANNELS, config.OUTPUT_CHANNELS, config.NGF, config.GEN_TYPE,
                                 norm=config.NORM_GEN, use_dropout=config.USE_DROPOUT_GEN,
                                 init_type=config.INIT_TYPE_GEN, init_gain=config.INIT_GAIN_GEN, gpu_ids=[], # DDP handles GPUs
                                 use_attention=config.USE_ATTENTION, attention_type=config.ATTENTION_TYPE)
    netG_B2A = networks.define_G(config.OUTPUT_CHANNELS, config.INPUT_CHANNELS, config.NGF, config.GEN_TYPE,
                                 norm=config.NORM_GEN, use_dropout=config.USE_DROPOUT_GEN,
                                 init_type=config.INIT_TYPE_GEN, init_gain=config.INIT_GAIN_GEN, gpu_ids=[],
                                 use_attention=config.USE_ATTENTION, attention_type=config.ATTENTION_TYPE)
    netD_A = networks.define_D(config.INPUT_CHANNELS, config.NDF, config.DISC_TYPE,
                               n_layers_D=config.N_LAYERS_DISC, norm=config.NORM_DISC,
                               init_type=config.INIT_TYPE_DISC, init_gain=config.INIT_GAIN_DISC, gpu_ids=[])
    netD_B = networks.define_D(config.OUTPUT_CHANNELS, config.NDF, config.DISC_TYPE,
                               n_layers_D=config.N_LAYERS_DISC, norm=config.NORM_DISC,
                               init_type=config.INIT_TYPE_DISC, init_gain=config.INIT_GAIN_DISC, gpu_ids=[])

    # Move models to the assigned device *before* DDP wrapping
    netG_A2B.to(device)
    netG_B2A.to(device)
    netD_A.to(device)
    netD_B.to(device)

    # --- 5. Loss Functions ---
    if config.LOSS_MODE == 'lsgan':
        criterionGAN = nn.MSELoss()
    elif config.LOSS_MODE == 'vanilla':
        criterionGAN = nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(f'Loss mode {config.LOSS_MODE} not implemented')

    criterionCycle = nn.L1Loss() # L1 loss for standard cycle consistency
    criterionIdt = nn.L1Loss()   # L1 loss for identity

    # --- VGG Loss Instantiation (Conditional) ---
    vgg_loss_module = None
    if use_vgg_loss:
        print(f"[Rank {rank}] Initializing VGG Perceptual Loss module...")
        try:
            # Determine which layers to use from config or use default
            vgg_layers = getattr(config, 'VGG_FEATURE_LAYERS', None) # Check if defined in config
            vgg_loss_module = VGG19PerceptualLoss(feature_layers=vgg_layers).to(device)
            vgg_loss_module.eval() # Ensure VGG is in eval mode and weights are frozen
            print(f"[Rank {rank}] VGG Loss module initialized and moved to {device}.")
        except Exception as e:
            print(f"[Rank {rank}] WARNING: Failed to initialize VGG Loss module: {e}. Disabling VGG Loss.")
            use_vgg_loss = False # Disable if initialization fails
            lambda_vgg = 0.0
    # ------------------------------------------

    # --- 6. Optimizers ---
    optimizer_G = optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                             lr=config.LR_G, betas=(config.BETA1, config.BETA2))
    optimizer_D = optim.Adam(itertools.chain(netD_A.parameters(), netD_B.parameters()),
                             lr=config.LR_D, betas=(config.BETA1, config.BETA2))

    # --- 7. Learning Rate Schedulers ---
    scheduler_G = get_scheduler(optimizer_G, config)
    scheduler_D = get_scheduler(optimizer_D, config)

    # --- 8. Image Pool ---
    fake_A_pool = ImagePool(config.POOL_SIZE)
    fake_B_pool = ImagePool(config.POOL_SIZE)

    # --- Initialize Loss Tracker (Only on main process) ---
    loss_tracker = None
    if is_main_process() and config.ENABLE_LOSS_TRACKING:
        loss_dir = os.path.join(config.OUTPUT_DIR, "loss_tracking")
        loss_tracker = LossTracker(loss_dir)
        print(f"[Rank {rank}] Loss tracking initialized. Saving to: {loss_dir}")
        print(f"[Rank {rank}] Loss plots will be generated every {config.LOSS_PLOT_FREQ} epochs")
        print(f"[Rank {rank}] Loss data will be saved every {config.LOSS_SAVE_FREQ} epochs")

    # --- AMP: GradScaler ---
    # Creates GradScalers once at the beginning of training.
    # enabled=False allows seamless disabling of AMP via config or if CUDA not available
    scaler_G = GradScaler(enabled=amp_enabled)
    scaler_D = GradScaler(enabled=amp_enabled)

    # --- 9. Checkpoint Loading (Resume) ---
    start_epoch = config.START_EPOCH
    if config.START_EPOCH > 1:
        latest_chkpt_path = os.path.join(config.CHECKPOINT_DIR, 'latest_checkpoint.pth.tar')
        # Corrected logic: checkpoint name should be START_EPOCH - 1
        epoch_chkpt_path = os.path.join(config.CHECKPOINT_DIR, f'{config.START_EPOCH - 1:03d}_checkpoint.pth.tar')

        load_path = None
        if os.path.exists(latest_chkpt_path):
             load_path = latest_chkpt_path
             print(f"[Rank {rank}] Found latest checkpoint: {load_path}. Attempting to load...")
        elif os.path.exists(epoch_chkpt_path):
             load_path = epoch_chkpt_path
             print(f"[Rank {rank}] Found epoch-specific checkpoint: {load_path}. Attempting to load...")
        else:
             print(f"[Rank {rank}] No checkpoint found to resume from epoch {config.START_EPOCH}. Training from scratch.")

        if load_path:
            try:
                # Pass scalers to load_checkpoint
                loaded_epoch = load_checkpoint(
                    load_path, netG_A2B, netG_B2A, netD_A, netD_B,
                    optimizer_G, optimizer_D, scheduler_G, scheduler_D,
                    device=device, # Pass target device for optimizer state
                    scaler_G=scaler_G, # Pass scaler instance G
                    scaler_D=scaler_D  # Pass scaler instance D
                )
                # Ensure scheduler's last_epoch and start_epoch are consistent
                start_epoch = loaded_epoch # Use epoch loaded from checkpoint
                scheduler_G.last_epoch = start_epoch - 1
                scheduler_D.last_epoch = start_epoch - 1
                config.START_EPOCH = start_epoch # Update config in case it differs
                print(f"[Rank {rank}] Checkpoint loaded successfully. Resuming from epoch {start_epoch}.")
            except Exception as e:
                print(f"[Rank {rank}] Error loading checkpoint {load_path}: {e}. Training from scratch.")
                start_epoch = 1 # Reset start epoch if loading failed

    # --- Wrap Models with DDP *after* loading state dict ---
    print(f"[Rank {rank}] Wrapping models with DDP...")
    # Consider find_unused_parameters=True if using complex conditional logic in models
    netG_A2B = DDP(netG_A2B, device_ids=([rank] if device.type == 'cuda' else None)) # Pass device_ids only for CUDA
    netG_B2A = DDP(netG_B2A, device_ids=([rank] if device.type == 'cuda' else None))
    netD_A = DDP(netD_A, device_ids=([rank] if device.type == 'cuda' else None))
    netD_B = DDP(netD_B, device_ids=([rank] if device.type == 'cuda' else None))
    print(f"[Rank {rank}] Models wrapped with DDP.")

    # --- 10. Training Loop ---
    total_batches = len(train_loader)
    print(f"\n[Rank {rank}] Starting training from epoch {start_epoch} to {config.NUM_EPOCHS}...")

    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        start_time_epoch = time.time()
        # Set epoch for sampler, ensures shuffling works correctly across epochs in DDP
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        # Use tqdm for progress bar only on the main process
        epoch_iterator = train_loader
        if is_main_process():
            epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS}", leave=False)

        # Accumulated losses for logging (main process only)
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        epoch_loss_GAN_A2B = 0.0
        epoch_loss_GAN_B2A = 0.0
        epoch_loss_cycle_A_L1 = 0.0
        epoch_loss_cycle_B_L1 = 0.0
        epoch_loss_cycle_A_vgg = 0.0
        epoch_loss_cycle_B_vgg = 0.0
        epoch_loss_idt_A = 0.0
        epoch_loss_idt_B = 0.0
        epoch_loss_vgg = 0.0 # Combined VGG loss for logging

        # Set models to train mode
        netG_A2B.train()
        netG_B2A.train()
        netD_A.train()
        netD_B.train()

        for i, batch in enumerate(epoch_iterator):
            # Move data to the correct device
            real_A = batch['A'].to(device, non_blocking=True) # LQ images
            real_B = batch['B'].to(device, non_blocking=True) # HQ images
            current_batch_size = real_A.size(0)

            # Adversarial ground truths (for LSGAN: 1.0 = real, 0.0 = fake)
            if config.LOSS_MODE == 'lsgan':
                target_real_label = 1.0
                target_fake_label = 0.0
            else: # vanilla GAN
                target_real_label = 1.0 # Will be converted to tensor of ones
                target_fake_label = 0.0 # Will be converted to tensor of zeros

            # ---------------------------------------------------
            # --- Generator Update (netG_A2B and netG_B2A) ---
            # ---------------------------------------------------
            optimizer_G.zero_grad(set_to_none=True) # set_to_none=True can improve performance

            # --- AMP: Apply autocast context ---
            with autocast(enabled=amp_enabled):
                # Identity Loss (Optional) - Calculated first for clarity
                loss_idt_A_val = 0.0
                loss_idt_B_val = 0.0
                if config.LAMBDA_IDENTITY > 0:
                    idt_B = netG_A2B(real_B)
                    loss_idt_B_val = criterionIdt(idt_B, real_B) * config.LAMBDA_IDENTITY
                    idt_A = netG_B2A(real_A)
                    loss_idt_A_val = criterionIdt(idt_A, real_A) * config.LAMBDA_IDENTITY

                # GAN Loss - Generate fake images and calculate adversarial loss
                fake_B = netG_A2B(real_A) # G_A2B(A) -> Fake B
                fake_A = netG_B2A(real_B) # G_B2A(B) -> Fake A

                # Loss G_A2B wants Discriminator D_B to classify fake_B as real
                pred_fake_B = netD_B(fake_B)
                if config.LOSS_MODE == 'lsgan':
                    target_tensor_real_G_B = torch.tensor(target_real_label, device=device).expand_as(pred_fake_B)
                else: # vanilla
                    target_tensor_real_G_B = torch.ones_like(pred_fake_B, device=device)
                loss_G_A2B_val = criterionGAN(pred_fake_B, target_tensor_real_G_B)

                # Loss G_B2A wants Discriminator D_A to classify fake_A as real
                pred_fake_A = netD_A(fake_A)
                if config.LOSS_MODE == 'lsgan':
                     target_tensor_real_G_A = torch.tensor(target_real_label, device=device).expand_as(pred_fake_A)
                else: # vanilla
                     target_tensor_real_G_A = torch.ones_like(pred_fake_A, device=device)
                loss_G_B2A_val = criterionGAN(pred_fake_A, target_tensor_real_G_A)

                # Cycle Consistency Loss (L1 Component)
                rec_A = netG_B2A(fake_B) # Reconstruct A: A -> fake_B -> rec_A
                loss_cycle_A_L1_val = criterionCycle(rec_A, real_A) * config.LAMBDA_CYCLE
                rec_B = netG_A2B(fake_A) # Reconstruct B: B -> fake_A -> rec_B
                loss_cycle_B_L1_val = criterionCycle(rec_B, real_B) * config.LAMBDA_CYCLE

                # --- VGG Perceptual Cycle Loss (Conditional) ---
                loss_cycle_A_vgg_val = 0.0
                loss_cycle_B_vgg_val = 0.0
                if vgg_loss_module is not None and lambda_vgg > 0:
                    # Features are extracted within the autocast context
                    features_rec_A = vgg_loss_module(rec_A)
                    features_real_A = vgg_loss_module(real_A)
                    loss_cycle_A_vgg_val = vgg_perceptual_criterion(vgg_loss_module, features_rec_A, features_real_A, criterion=criterionCycle) * lambda_vgg # Use L1 on features by default

                    features_rec_B = vgg_loss_module(rec_B)
                    features_real_B = vgg_loss_module(real_B)
                    loss_cycle_B_vgg_val = vgg_perceptual_criterion(vgg_loss_module, features_rec_B, features_real_B, criterion=criterionCycle) * lambda_vgg
                # ---------------------------------------------

                # Combined Generator Loss
                # Weights are applied to individual components before summing
                loss_G = (loss_G_A2B_val + loss_G_B2A_val +
                          loss_cycle_A_L1_val + loss_cycle_B_L1_val +
                          loss_cycle_A_vgg_val + loss_cycle_B_vgg_val + # Add VGG loss component
                          loss_idt_A_val + loss_idt_B_val)

            # --- End of autocast for G ---

            # --- AMP: Scale loss, backward, and update G ---
            scaler_G.scale(loss_G).backward()

            # --- Gradient Clipping (Uncommented) ---
            scaler_G.unscale_(optimizer_G) # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), max_norm=1.0)
            # ---------------------------------------

            scaler_G.step(optimizer_G)
            scaler_G.update()
            # ---------------------------------------------

            # -----------------------------------------------------
            # --- Discriminator Update (netD_A and netD_B) ---
            # -----------------------------------------------------
            optimizer_D.zero_grad(set_to_none=True)

            # --- AMP: Apply autocast context for D ---
            with autocast(enabled=amp_enabled):
                # --- Discriminator A Loss ---
                # Real A Loss
                pred_real_A = netD_A(real_A)
                if config.LOSS_MODE == 'lsgan':
                     target_tensor_real_D_A = torch.tensor(target_real_label, device=device).expand_as(pred_real_A)
                else: # vanilla
                     target_tensor_real_D_A = torch.ones_like(pred_real_A, device=device)
                loss_D_real_A = criterionGAN(pred_real_A, target_tensor_real_D_A)

                # Fake A Loss (using pooled fake_A and detaching)
                fake_A_pooled = fake_A_pool.query(fake_A)
                pred_fake_A_pool = netD_A(fake_A_pooled.detach()) # Use detached fake images
                if config.LOSS_MODE == 'lsgan':
                    target_tensor_fake_D_A = torch.tensor(target_fake_label, device=device).expand_as(pred_fake_A_pool)
                else: # vanilla
                    target_tensor_fake_D_A = torch.zeros_like(pred_fake_A_pool, device=device)
                loss_D_fake_A = criterionGAN(pred_fake_A_pool, target_tensor_fake_D_A)

                # Combine D_A losses
                loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5

                # --- Discriminator B Loss ---
                # Real B loss
                pred_real_B = netD_B(real_B)
                if config.LOSS_MODE == 'lsgan':
                    target_tensor_real_D_B = torch.tensor(target_real_label, device=device).expand_as(pred_real_B)
                else: # vanilla
                    target_tensor_real_D_B = torch.ones_like(pred_real_B, device=device)
                loss_D_real_B = criterionGAN(pred_real_B, target_tensor_real_D_B)

                # Fake B loss (using pooled fake_B and detaching)
                fake_B_pooled = fake_B_pool.query(fake_B)
                pred_fake_B_pool = netD_B(fake_B_pooled.detach()) # Use detached fake images
                if config.LOSS_MODE == 'lsgan':
                    target_tensor_fake_D_B = torch.tensor(target_fake_label, device=device).expand_as(pred_fake_B_pool)
                else: # vanilla
                    target_tensor_fake_D_B = torch.zeros_like(pred_fake_B_pool, device=device)
                loss_D_fake_B = criterionGAN(pred_fake_B_pool, target_tensor_fake_D_B)

                # Combine D_B losses
                loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5

            # --- End of autocast for D ---

            # --- AMP: Scale losses, backward (separately), and update D ---
            # Backward pass for D_A
            scaler_D.scale(loss_D_A).backward()
            # Backward pass for D_B
            scaler_D.scale(loss_D_B).backward()

            # --- Gradient Clipping (Uncommented) ---
            scaler_D.unscale_(optimizer_D) # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(itertools.chain(netD_A.parameters(), netD_B.parameters()), max_norm=1.0)
            # ---------------------------------------

            # Update Discriminator weights
            scaler_D.step(optimizer_D)
            scaler_D.update()
            # ---------------------------------------------

            # --- Logging (Only on main process) ---
            if is_main_process():
                # Accumulate losses for epoch average logging
                epoch_loss_G += loss_G.item()
                epoch_loss_D += (loss_D_A.item() + loss_D_B.item()) # Sum of both D losses
                epoch_loss_GAN_A2B += loss_G_A2B_val.item()
                epoch_loss_GAN_B2A += loss_G_B2A_val.item()
                epoch_loss_cycle_A_L1 += loss_cycle_A_L1_val.item()
                epoch_loss_cycle_B_L1 += loss_cycle_B_L1_val.item()
                epoch_loss_idt_A += loss_idt_A_val.item() if isinstance(loss_idt_A_val, torch.Tensor) else loss_idt_A_val
                epoch_loss_idt_B += loss_idt_B_val.item() if isinstance(loss_idt_B_val, torch.Tensor) else loss_idt_B_val

                current_loss_vgg = 0.0
                current_loss_vgg_A = 0.0
                current_loss_vgg_B = 0.0
                if use_vgg_loss:
                    current_loss_vgg_A = loss_cycle_A_vgg_val.item() if isinstance(loss_cycle_A_vgg_val, torch.Tensor) else loss_cycle_A_vgg_val
                    current_loss_vgg_B = loss_cycle_B_vgg_val.item() if isinstance(loss_cycle_B_vgg_val, torch.Tensor) else loss_cycle_B_vgg_val
                    current_loss_vgg = current_loss_vgg_A + current_loss_vgg_B
                    epoch_loss_vgg += current_loss_vgg
                    epoch_loss_cycle_A_vgg += current_loss_vgg_A
                    epoch_loss_cycle_B_vgg += current_loss_vgg_B

                # --- Add batch loss tracking ---
                if loss_tracker is not None:
                    current_iter = epoch * total_batches + i + 1
                    batch_losses = {
                        'loss_G': loss_G.item(),
                        'loss_D': loss_D_A.item() + loss_D_B.item(),
                        'loss_GAN_A2B': loss_G_A2B_val.item(),
                        'loss_GAN_B2A': loss_G_B2A_val.item(),
                        'loss_cycle_A_L1': loss_cycle_A_L1_val.item(),
                        'loss_cycle_B_L1': loss_cycle_B_L1_val.item(),
                        'loss_cycle_A_vgg': current_loss_vgg_A,
                        'loss_cycle_B_vgg': current_loss_vgg_B,
                        'loss_idt_A': loss_idt_A_val.item() if isinstance(loss_idt_A_val, torch.Tensor) else loss_idt_A_val,
                        'loss_idt_B': loss_idt_B_val.item() if isinstance(loss_idt_B_val, torch.Tensor) else loss_idt_B_val
                    }
                    loss_tracker.add_batch_loss(current_iter, epoch, i + 1, batch_losses)

                # Print log message periodically
                if (i + 1) % config.LOG_FREQ == 0:
                    current_iter = epoch * total_batches + i + 1
                    # Calculate batch averages carefully, avoiding division by zero
                    batches_done = i + 1
                    avg_loss_G_batch = epoch_loss_G / batches_done if batches_done > 0 else 0.0
                    avg_loss_D_batch = epoch_loss_D / batches_done if batches_done > 0 else 0.0
                    scale_G_val = scaler_G.get_scale()
                    scale_D_val = scaler_D.get_scale()
                    log_msg = (
                        f"\nEpoch [{epoch}/{config.NUM_EPOCHS}], Batch [{i+1}/{total_batches}], "
                        f"Loss G: {loss_G.item():.4f} (Avg: {avg_loss_G_batch:.4f}), "
                        f"Loss D: {(loss_D_A.item() + loss_D_B.item()):.4f} (Avg: {avg_loss_D_batch:.4f}), \n"
                        f"  GAN: G_A2B={loss_G_A2B_val.item():.4f}, G_B2A={loss_G_B2A_val.item():.4f} | "
                        f"Cycle(L1): A={loss_cycle_A_L1_val.item():.4f}, B={loss_cycle_B_L1_val.item():.4f} | "
                    )
                    if use_vgg_loss:
                        log_msg += f"Cycle(VGG): A={current_loss_vgg_A:.4f}, B={current_loss_vgg_B:.4f} | "
                    log_msg += (
                        f"Idt: A={loss_idt_A_val.item() if isinstance(loss_idt_A_val, torch.Tensor) else loss_idt_A_val:.4f}, "
                        f"B={loss_idt_B_val.item() if isinstance(loss_idt_B_val, torch.Tensor) else loss_idt_B_val:.4f} | "
                        f"Scale: G={scale_G_val:.1f}, D={scale_D_val:.1f}"
                    )
                    # Check for NaN/Inf in averages before printing
                    if not np.isfinite(avg_loss_G_batch) or not np.isfinite(avg_loss_D_batch):
                         print(f"\nWarning: NaN/Inf detected in average losses at batch {i+1}. Current losses: G={loss_G.item()}, D_A={loss_D_A.item()}, D_B={loss_D_B.item()}")
                    else:
                         print(log_msg)


                # Save 'latest' checkpoint periodically (optional)
                if config.SAVE_LATEST_FREQ > 0 and (i + 1) % config.SAVE_LATEST_FREQ == 0:
                    # Ensure state includes scaler states if AMP is enabled
                    chkpt_state = {
                        'epoch': epoch, # Save current epoch, will resume from epoch + 1
                        'netG_A2B_state_dict': netG_A2B.module.state_dict(), # Use .module with DDP
                        'netG_B2A_state_dict': netG_B2A.module.state_dict(),
                        'netD_A_state_dict': netD_A.module.state_dict(),
                        'netD_B_state_dict': netD_B.module.state_dict(),
                        'optimizer_G_state_dict': optimizer_G.state_dict(),
                        'optimizer_D_state_dict': optimizer_D.state_dict(),
                        'scheduler_G_state_dict': scheduler_G.state_dict(),
                        'scheduler_D_state_dict': scheduler_D.state_dict()
                    }
                    if amp_enabled:
                        chkpt_state['scaler_G_state_dict'] = scaler_G.state_dict()
                        chkpt_state['scaler_D_state_dict'] = scaler_D.state_dict()

                    save_checkpoint(chkpt_state, filename="latest_checkpoint.pth.tar", directory=config.CHECKPOINT_DIR)

        # --- End of Epoch ---
        end_time_epoch = time.time()
        epoch_duration = end_time_epoch - start_time_epoch

        # Update learning rate schedulers (after the epoch is fully completed)
        scheduler_G.step()
        scheduler_D.step()

        # --- Epoch Logging and Saving (Only on main process) ---
        if is_main_process():
            # Calculate averages for the completed epoch
            avg_loss_G_epoch = epoch_loss_G / total_batches if total_batches > 0 else 0.0
            avg_loss_D_epoch = epoch_loss_D / total_batches if total_batches > 0 else 0.0
            avg_loss_GAN_A2B_epoch = epoch_loss_GAN_A2B / total_batches if total_batches > 0 else 0.0
            avg_loss_GAN_B2A_epoch = epoch_loss_GAN_B2A / total_batches if total_batches > 0 else 0.0
            avg_loss_cycle_A_L1_epoch = epoch_loss_cycle_A_L1 / total_batches if total_batches > 0 else 0.0
            avg_loss_cycle_B_L1_epoch = epoch_loss_cycle_B_L1 / total_batches if total_batches > 0 else 0.0
            avg_loss_idt_A_epoch = epoch_loss_idt_A / total_batches if total_batches > 0 else 0.0
            avg_loss_idt_B_epoch = epoch_loss_idt_B / total_batches if total_batches > 0 else 0.0
            avg_loss_vgg_epoch = epoch_loss_vgg / total_batches if use_vgg_loss and total_batches > 0 else 0.0
            avg_loss_cycle_A_vgg_epoch = epoch_loss_cycle_A_vgg / total_batches if use_vgg_loss and total_batches > 0 else 0.0
            avg_loss_cycle_B_vgg_epoch = epoch_loss_cycle_B_vgg / total_batches if use_vgg_loss and total_batches > 0 else 0.0


            # Get current learning rates and scaler states
            current_lr_G = optimizer_G.param_groups[0]['lr']
            current_lr_D = optimizer_D.param_groups[0]['lr']
            scale_G_val = scaler_G.get_scale()
            scale_D_val = scaler_D.get_scale()

            print("-" * 80)
            print(f"End of Epoch [{epoch}/{config.NUM_EPOCHS}] - Time: {epoch_duration:.2f}s")
            # Check for NaN/Inf before printing epoch summary
            if not np.isfinite(avg_loss_G_epoch) or not np.isfinite(avg_loss_D_epoch):
                print("  *** NaN/Inf detected in average losses for the epoch! ***")
            else:
                print(f"  Avg Loss G: {avg_loss_G_epoch:.4f}, Avg Loss D: {avg_loss_D_epoch:.4f}")
                print(f"  Avg G_A2B: {avg_loss_GAN_A2B_epoch:.4f}, Avg G_B2A: {avg_loss_GAN_B2A_epoch:.4f}")
                print(f"  Avg Cycle L1: A={avg_loss_cycle_A_L1_epoch:.4f}, B={avg_loss_cycle_B_L1_epoch:.4f}")
                if use_vgg_loss:
                    print(f"  Avg Cycle VGG: {avg_loss_vgg_epoch:.4f} (A={avg_loss_cycle_A_vgg_epoch:.4f}, B={avg_loss_cycle_B_vgg_epoch:.4f})")
                print(f"  Avg Idt: A={avg_loss_idt_A_epoch:.4f}, B={avg_loss_idt_B_epoch:.4f}")
            print(f"  Current LR G: {current_lr_G:.6f}, Current LR D: {current_lr_D:.6f}")
            if amp_enabled:
                 print(f"  Final Scale G: {scale_G_val:.1f}, Final Scale D: {scale_D_val:.1f}")
            print("-" * 80)

            # --- Add epoch loss tracking and plot generation ---
            if loss_tracker is not None:
                epoch_losses = {
                    'loss_G': avg_loss_G_epoch,
                    'loss_D': avg_loss_D_epoch,
                    'loss_GAN_A2B': avg_loss_GAN_A2B_epoch,
                    'loss_GAN_B2A': avg_loss_GAN_B2A_epoch,
                    'loss_cycle_A_L1': avg_loss_cycle_A_L1_epoch,
                    'loss_cycle_B_L1': avg_loss_cycle_B_L1_epoch,
                    'loss_cycle_A_vgg': avg_loss_cycle_A_vgg_epoch,
                    'loss_cycle_B_vgg': avg_loss_cycle_B_vgg_epoch,
                    'loss_idt_A': avg_loss_idt_A_epoch,
                    'loss_idt_B': avg_loss_idt_B_epoch,
                    'loss_vgg_total': avg_loss_vgg_epoch
                }
                loss_tracker.add_epoch_loss(epoch, epoch_losses, current_lr_G, current_lr_D, epoch_duration)
                
                # Save losses and generate plots every few epochs or at the end
                if epoch % config.LOSS_SAVE_FREQ == 0 or epoch == config.NUM_EPOCHS:
                    try:
                        loss_tracker.save_losses()
                        print(f"  Loss data saved for epoch {epoch}")
                    except Exception as e:
                        print(f"  Warning: Failed to save loss data: {e}")
                
                if epoch % config.LOSS_PLOT_FREQ == 0 or epoch == config.NUM_EPOCHS:
                    try:
                        loss_tracker.plot_losses()
                        print(f"  Loss plots generated for epoch {epoch}")
                    except Exception as e:
                        print(f"  Warning: Failed to generate loss plots: {e}")

            # Save model checkpoint at specified frequency or at the end
            if epoch % config.SAVE_EPOCH_FREQ == 0 or epoch == config.NUM_EPOCHS:
                chkpt_state = {
                    'epoch': epoch + 1, # Save as next epoch to start from
                    'netG_A2B_state_dict': netG_A2B.module.state_dict(),
                    'netG_B2A_state_dict': netG_B2A.module.state_dict(),
                    'netD_A_state_dict': netD_A.module.state_dict(),
                    'netD_B_state_dict': netD_B.module.state_dict(),
                    'optimizer_G_state_dict': optimizer_G.state_dict(),
                    'optimizer_D_state_dict': optimizer_D.state_dict(),
                    'scheduler_G_state_dict': scheduler_G.state_dict(),
                    'scheduler_D_state_dict': scheduler_D.state_dict()
                }
                if amp_enabled:
                    chkpt_state['scaler_G_state_dict'] = scaler_G.state_dict()
                    chkpt_state['scaler_D_state_dict'] = scaler_D.state_dict()

                chkpt_filename = f"{epoch:03d}_checkpoint.pth.tar"
                save_checkpoint(chkpt_state, filename=chkpt_filename, directory=config.CHECKPOINT_DIR)

                # Also save individual model weights for easier evaluation loading
                try:
                    torch.save(netG_A2B.module.state_dict(), os.path.join(config.CHECKPOINT_DIR, f'{epoch:03d}_netG_A2B.pth'))
                except Exception as e:
                    print(f"Warning: Failed to save {epoch:03d}_netG_A2B.pth: {e}")
                try:
                    torch.save(netG_B2A.module.state_dict(), os.path.join(config.CHECKPOINT_DIR, f'{epoch:03d}_netG_B2A.pth'))
                except Exception as e:
                    print(f"Warning: Failed to save {epoch:03d}_netG_B2A.pth: {e}")

                # Save latest models too (overwrite previous latest)
                try:
                    torch.save(netG_A2B.module.state_dict(), os.path.join(config.CHECKPOINT_DIR, f'latest_netG_A2B.pth'))
                except Exception as e:
                    print(f"Warning: Failed to save latest_netG_A2B.pth: {e}")
                try:
                    torch.save(netG_B2A.module.state_dict(), os.path.join(config.CHECKPOINT_DIR, f'latest_netG_B2A.pth'))
                except Exception as e:
                    print(f"Warning: Failed to save latest_netG_B2A.pth: {e}")


            # Save visual samples at specified frequency
            if epoch % config.SAVE_SAMPLES_EPOCH_FREQ == 0:
                # Get the last processed batch from the main process for visualization
                sample_batch = {'A': real_A.cpu(), 'B': real_B.cpu()} # Move to CPU before passing to save function if needed
                try:
                     # Pass the underlying models (.module) to save_sample_images
                     # The save function expects tensors on the correct device or handles CPU tensors
                     save_sample_images(epoch, {'A': real_A, 'B': real_B}, netG_A2B.module, netG_B2A.module, device, max_samples=4)
                except Exception as e:
                     print(f"\nWarning: Failed to save sample images for epoch {epoch}. Error: {e}")


    # --- End of Training ---
    print(f"[Rank {rank}] Training finished.")
    cleanup_ddp()


if __name__ == "__main__":
    # --- Argument Parsing for DDP launch ---
    # torchrun automatically sets RANK, LOCAL_RANK, WORLD_SIZE env vars.
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) # Often equivalent to rank if 1 node

    print(f"Starting training script. World Size: {world_size}, Global Rank: {rank}, Local Rank: {local_rank}")

    if world_size > 1 and not dist.is_available():
         raise RuntimeError("Distributed training requested but Distributed module not available.")

    # Check CUDA availability more robustly
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        num_gpus = torch.cuda.device_count()
        print(f"CUDA is available. Number of GPUs: {num_gpus}")
        if world_size > num_gpus:
             print(f"Warning: WORLD_SIZE ({world_size}) > available GPUs ({num_gpus}). Some processes might idle or share GPUs.")
        if local_rank >= num_gpus:
             raise RuntimeError(f"LOCAL_RANK {local_rank} is invalid for available GPUs ({num_gpus})")
    else:
        print("Warning: CUDA not available.")
        if world_size > 1:
            print("Distributed training requested but no CUDA GPUs available. Attempting DDP on CPU using 'gloo' backend.")
        else:
             print("Running on CPU (single process).")

    # Start the main training function
    try:
        main(rank, world_size)
    except Exception as e:
        print(f"[Rank {rank}] Exception during training: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        # Ensure DDP cleanup happens even if main function crashes
        cleanup_ddp()
        # It's important to exit with a non-zero code if an error occurred
        exit(1) # Exit script to signal failure to torchrun/user
