# src/train.py (con AMP integrado)

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

# Import project modules
from src import config
from src.dataset import NpzImageDataset
from src.models import networks
from src.utils import (
    ImagePool, get_scheduler, init_weights, save_checkpoint, load_checkpoint,
    save_sample_images, setup_ddp, cleanup_ddp, is_main_process, tensor2im
)

def main(rank, world_size):
    """Main training function with AMP."""

    # --- 1. DDP Setup ---
    print(f"Initializing DDP for Rank {rank}/{world_size}...")
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}") # Pin to specific GPU

    # --- 2. Configuration and Paths (Accessible via config) ---
    # Check if AMP is enabled in config (add AMP_ENABLED = True/False to config.py)
    amp_enabled = getattr(config, 'AMP_ENABLED', False) # Default to False if not defined
    if amp_enabled:
        print(f"[Rank {rank}] Automatic Mixed Precision (AMP) ENABLED.")
    else:
        print(f"[Rank {rank}] Automatic Mixed Precision (AMP) DISABLED.")


    # --- 3. Dataset and DataLoader ---
    print(f"[Rank {rank}] Loading dataset...")
    train_dataset = NpzImageDataset(npz_path=config.TRAIN_DATA_PATH, mode='train')

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
        pin_memory=True, # Usually improves performance by speeding up CPU->GPU transfer
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

    # Move models to the assigned GPU device *before* DDP wrapping
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

    criterionCycle = nn.L1Loss() # Cycle consistency loss
    criterionIdt = nn.L1Loss()   # Identity loss

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

    # --- AMP: GradScaler ---
    # Creates a GradScaler once at the beginning of training.
    # enabled=False allows seamless disabling of AMP via config
    scaler = GradScaler(enabled=amp_enabled)
    scaler_G = GradScaler(enabled=amp_enabled) # Using separate scalers can sometimes be beneficial
    scaler_D = GradScaler(enabled=amp_enabled) # if updates are fully separate, but one often works fine. Let's use two for clarity.


    # --- 9. Checkpoint Loading (Resume) ---
    start_epoch = config.START_EPOCH
    # Try loading the 'latest' checkpoint by default if START_EPOCH suggests resuming
    # All processes load the same checkpoint.
    if config.START_EPOCH > 1:
        latest_chkpt_path = os.path.join(config.CHECKPOINT_DIR, 'latest_checkpoint.pth.tar')
        epoch_chkpt_path = os.path.join(config.CHECKPOINT_DIR, f'{config.START_EPOCH - 1:03d}_checkpoint.pth.tar')

        load_path = None
        if os.path.exists(latest_chkpt_path):
             load_path = latest_chkpt_path
             print(f"[Rank {rank}] Found latest checkpoint: {load_path}. Attempting to load...")
        elif os.path.exists(epoch_chkpt_path):
             load_path = epoch_chkpt_path
             print(f"[Rank {rank}] Found epoch-specific checkpoint: {load_path}. Attempting to load...")
        else:
             print(f"[Rank {rank}] No checkpoint found to resume from epoch {config.START_EPOCH}. Starting from scratch.")

        if load_path:
            try:
                # --- AMP: Pass scalers to load_checkpoint ---
                start_epoch = load_checkpoint(
                    load_path, netG_A2B, netG_B2A, netD_A, netD_B,
                    optimizer_G, optimizer_D, scheduler_G, scheduler_D,
                    device=device,
                    scaler_G=scaler_G, # Pass scaler instance G
                    scaler_D=scaler_D  # Pass scaler instance D
                )
                # -------------------------------------------
                scheduler_G.last_epoch = start_epoch - 1
                scheduler_D.last_epoch = start_epoch - 1
                config.START_EPOCH = start_epoch
                print(f"[Rank {rank}] Checkpoint loaded successfully. Resuming from epoch {start_epoch}.")
            except Exception as e:
                print(f"[Rank {rank}] Error loading checkpoint {load_path}: {e}. Training from scratch.")
                start_epoch = 1

    # --- Wrap Models with DDP *after* loading state dict ---
    print(f"[Rank {rank}] Wrapping models with DDP...")
    netG_A2B = DDP(netG_A2B, device_ids=[rank], find_unused_parameters=False) # Set find_unused_parameters based on potential conditional logic in models
    netG_B2A = DDP(netG_B2A, device_ids=[rank], find_unused_parameters=False)
    netD_A = DDP(netD_A, device_ids=[rank], find_unused_parameters=False)
    netD_B = DDP(netD_B, device_ids=[rank], find_unused_parameters=False)
    print(f"[Rank {rank}] Models wrapped with DDP.")

    # --- 10. Training Loop ---
    total_batches = len(train_loader)
    print(f"\n[Rank {rank}] Starting training from epoch {start_epoch} to {config.NUM_EPOCHS}...")

    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        start_time_epoch = time.time()
        train_sampler.set_epoch(epoch)

        epoch_iterator = train_loader
        if is_main_process():
            epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS}", leave=False)

        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        epoch_loss_GAN_A2B = 0.0
        epoch_loss_GAN_B2A = 0.0
        epoch_loss_cycle_A = 0.0
        epoch_loss_cycle_B = 0.0
        epoch_loss_idt_A = 0.0
        epoch_loss_idt_B = 0.0

        for i, batch in enumerate(epoch_iterator):
            real_A = batch['A'].to(device, non_blocking=True) # LQ images
            real_B = batch['B'].to(device, non_blocking=True) # HQ images
            current_batch_size = real_A.size(0)

            if config.LOSS_MODE == 'lsgan':
                target_real_label = 1.0
                target_fake_label = 0.0
            else:
                target_real_label = 1.0
                target_fake_label = 0.0

            # --- Generator Update (netG_A2B and netG_B2A) ---
            optimizer_G.zero_grad(set_to_none=True) # set_to_none=True can improve performance

            # --- AMP: Apply autocast context ---
            with autocast(enabled=amp_enabled):
                # Identity Loss (Optional)
                loss_idt_A = 0.0
                loss_idt_B = 0.0
                if config.LAMBDA_IDENTITY > 0:
                    idt_B = netG_A2B(real_B)
                    loss_idt_B = criterionIdt(idt_B, real_B) * config.LAMBDA_CYCLE * config.LAMBDA_IDENTITY
                    idt_A = netG_B2A(real_A)
                    loss_idt_A = criterionIdt(idt_A, real_A) * config.LAMBDA_CYCLE * config.LAMBDA_IDENTITY

                # GAN Loss
                fake_B = netG_A2B(real_A)
                fake_A = netG_B2A(real_B)

                pred_fake_B = netD_B(fake_B)
                if config.LOSS_MODE == 'lsgan':
                    target_tensor_real_G_B = torch.tensor(target_real_label).expand_as(pred_fake_B).to(device)
                else:
                    target_tensor_real_G_B = torch.ones_like(pred_fake_B).to(device)
                loss_G_A2B = criterionGAN(pred_fake_B, target_tensor_real_G_B)

                pred_fake_A = netD_A(fake_A)
                if config.LOSS_MODE == 'lsgan':
                     target_tensor_real_G_A = torch.tensor(target_real_label).expand_as(pred_fake_A).to(device)
                else:
                     target_tensor_real_G_A = torch.ones_like(pred_fake_A).to(device)
                loss_G_B2A = criterionGAN(pred_fake_A, target_tensor_real_G_A)

                # Cycle Consistency Loss
                rec_A = netG_B2A(fake_B)
                loss_cycle_A = criterionCycle(rec_A, real_A) * config.LAMBDA_CYCLE
                rec_B = netG_A2B(fake_A)
                loss_cycle_B = criterionCycle(rec_B, real_B) * config.LAMBDA_CYCLE

                # Combined Generator Loss
                loss_G = loss_G_A2B + loss_G_B2A + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
            # --- End of autocast for G ---

            # --- AMP: Scale loss, backward, and update G ---
            scaler_G.scale(loss_G).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()
            # ---------------------------------------------

            # --- Discriminator Update (netD_A and netD_B) ---
            optimizer_D.zero_grad(set_to_none=True)

            # --- AMP: Apply autocast context ---
            with autocast(enabled=amp_enabled):
                # Discriminator A Loss
                pred_real_A = netD_A(real_A)
                if config.LOSS_MODE == 'lsgan':
                     target_tensor_real_D_A = torch.tensor(target_real_label).expand_as(pred_real_A).to(device)
                else:
                     target_tensor_real_D_A = torch.ones_like(pred_real_A).to(device)
                loss_D_real_A = criterionGAN(pred_real_A, target_tensor_real_D_A)

                fake_A_pooled = fake_A_pool.query(fake_A)
                pred_fake_A_pool = netD_A(fake_A_pooled.detach()) # Use detached fake images
                if config.LOSS_MODE == 'lsgan':
                    target_tensor_fake_D_A = torch.tensor(target_fake_label).expand_as(pred_fake_A_pool).to(device)
                else:
                    target_tensor_fake_D_A = torch.zeros_like(pred_fake_A_pool).to(device)
                loss_D_fake_A = criterionGAN(pred_fake_A_pool, target_tensor_fake_D_A)
                loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5

                # Discriminator B Loss
                pred_real_B = netD_B(real_B)
                if config.LOSS_MODE == 'lsgan':
                    target_tensor_real_D_B = torch.tensor(target_real_label).expand_as(pred_real_B).to(device)
                else:
                    target_tensor_real_D_B = torch.ones_like(pred_real_B).to(device)
                loss_D_real_B = criterionGAN(pred_real_B, target_tensor_real_D_B)

                fake_B_pooled = fake_B_pool.query(fake_B)
                pred_fake_B_pool = netD_B(fake_B_pooled.detach()) # Use detached fake images
                if config.LOSS_MODE == 'lsgan':
                    target_tensor_fake_D_B = torch.tensor(target_fake_label).expand_as(pred_fake_B_pool).to(device)
                else:
                    target_tensor_fake_D_B = torch.zeros_like(pred_fake_B_pool).to(device)
                loss_D_fake_B = criterionGAN(pred_fake_B_pool, target_tensor_fake_D_B)
                loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5

            # --- End of autocast for D ---

            # --- AMP: Scale losses, backward (separately?), and update D ---
            # Option 1: Combine D losses and scale once (requires one scaler)
            # loss_D_total = loss_D_A + loss_D_B
            # scaler_D.scale(loss_D_total).backward()

            # Option 2: Scale and backward separately (requires separate scalers or careful handling)
            # This might be slightly more numerically stable if losses have very different magnitudes
            scaler_D.scale(loss_D_A).backward()
            scaler_D.scale(loss_D_B).backward()

            scaler_D.step(optimizer_D)
            scaler_D.update()
            # ---------------------------------------------

            # --- Logging (Only on main process) ---
            if is_main_process():
                epoch_loss_G += loss_G.item()
                epoch_loss_D += (loss_D_A.item() + loss_D_B.item())
                epoch_loss_GAN_A2B += loss_G_A2B.item()
                epoch_loss_GAN_B2A += loss_G_B2A.item()
                epoch_loss_cycle_A += loss_cycle_A.item()
                epoch_loss_cycle_B += loss_cycle_B.item()
                epoch_loss_idt_A += loss_idt_A.item() if isinstance(loss_idt_A, torch.Tensor) else loss_idt_A
                epoch_loss_idt_B += loss_idt_B.item() if isinstance(loss_idt_B, torch.Tensor) else loss_idt_B

                if (i + 1) % config.LOG_FREQ == 0:
                    current_iter = epoch * total_batches + i + 1
                    avg_loss_G = epoch_loss_G / (i + 1)
                    avg_loss_D = epoch_loss_D / (i + 1)
                    # --- AMP: Log scaler state (optional) ---
                    scale_G_val = scaler_G.get_scale()
                    scale_D_val = scaler_D.get_scale()
                    print(f"\nEpoch [{epoch}/{config.NUM_EPOCHS}], Batch [{i+1}/{total_batches}], "
                          f"Loss G: {loss_G.item():.4f} (Avg: {avg_loss_G:.4f}), "
                          f"Loss D: {(loss_D_A.item() + loss_D_B.item()):.4f} (Avg: {avg_loss_D:.4f}), "
                          f"G_A2B: {loss_G_A2B.item():.4f}, G_B2A: {loss_G_B2A.item():.4f}, "
                          f"Cycle A: {loss_cycle_A.item():.4f}, Cycle B: {loss_cycle_B.item():.4f}, "
                          f"Idt A: {loss_idt_A.item() if isinstance(loss_idt_A, torch.Tensor) else loss_idt_A:.4f}, "
                          f"Idt B: {loss_idt_B.item() if isinstance(loss_idt_B, torch.Tensor) else loss_idt_B:.4f}, "
                          f"Scale G: {scale_G_val:.1f}, Scale D: {scale_D_val:.1f}") # Log scale values
                    # ----------------------------------------

                if config.SAVE_LATEST_FREQ > 0 and (i + 1) % config.SAVE_LATEST_FREQ == 0:
                    chkpt_state = {
                        'epoch': epoch,
                        'netG_A2B_state_dict': netG_A2B.module.state_dict(),
                        'netG_B2A_state_dict': netG_B2A.module.state_dict(),
                        'netD_A_state_dict': netD_A.module.state_dict(),
                        'netD_B_state_dict': netD_B.module.state_dict(),
                        'optimizer_G_state_dict': optimizer_G.state_dict(),
                        'optimizer_D_state_dict': optimizer_D.state_dict(),
                        'scheduler_G_state_dict': scheduler_G.state_dict(),
                        'scheduler_D_state_dict': scheduler_D.state_dict(),
                        # --- AMP: Save scaler states ---
                        'scaler_G_state_dict': scaler_G.state_dict(),
                        'scaler_D_state_dict': scaler_D.state_dict()
                        # -------------------------------
                    }
                    save_checkpoint(chkpt_state, filename="latest_checkpoint.pth.tar", directory=config.CHECKPOINT_DIR)

        # --- End of Epoch ---
        end_time_epoch = time.time()
        epoch_duration = end_time_epoch - start_time_epoch

        scheduler_G.step()
        scheduler_D.step()

        if is_main_process():
            avg_loss_G_epoch = epoch_loss_G / total_batches
            avg_loss_D_epoch = epoch_loss_D / total_batches
            current_lr_G = optimizer_G.param_groups[0]['lr']
            current_lr_D = optimizer_D.param_groups[0]['lr']

            print("-" * 80)
            print(f"End of Epoch [{epoch}/{config.NUM_EPOCHS}] - Time: {epoch_duration:.2f}s")
            print(f"  Avg Loss G: {avg_loss_G_epoch:.4f}, Avg Loss D: {avg_loss_D_epoch:.4f}")
            print(f"  Avg G_A2B: {epoch_loss_GAN_A2B / total_batches:.4f}, Avg G_B2A: {epoch_loss_GAN_B2A / total_batches:.4f}")
            print(f"  Avg Cycle A: {epoch_loss_cycle_A / total_batches:.4f}, Avg Cycle B: {epoch_loss_cycle_B / total_batches:.4f}")
            print(f"  Avg Idt A: {epoch_loss_idt_A / total_batches:.4f}, Avg Idt B: {epoch_loss_idt_B / total_batches:.4f}")
            print(f"  Current LR G: {current_lr_G:.6f}, Current LR D: {current_lr_D:.6f}")
            # --- AMP: Log final scale for epoch ---
            scale_G_val = scaler_G.get_scale()
            scale_D_val = scaler_D.get_scale()
            print(f"  Final Scale G: {scale_G_val:.1f}, Final Scale D: {scale_D_val:.1f}")
            print("-" * 80)
            # ------------------------------------

            if epoch % config.SAVE_EPOCH_FREQ == 0 or epoch == config.NUM_EPOCHS:
                chkpt_state = {
                    'epoch': epoch + 1,
                    'netG_A2B_state_dict': netG_A2B.module.state_dict(),
                    'netG_B2A_state_dict': netG_B2A.module.state_dict(),
                    'netD_A_state_dict': netD_A.module.state_dict(),
                    'netD_B_state_dict': netD_B.module.state_dict(),
                    'optimizer_G_state_dict': optimizer_G.state_dict(),
                    'optimizer_D_state_dict': optimizer_D.state_dict(),
                    'scheduler_G_state_dict': scheduler_G.state_dict(),
                    'scheduler_D_state_dict': scheduler_D.state_dict(),
                    # --- AMP: Save scaler states ---
                    'scaler_G_state_dict': scaler_G.state_dict(),
                    'scaler_D_state_dict': scaler_D.state_dict()
                    # -------------------------------
                }
                chkpt_filename = f"{epoch:03d}_checkpoint.pth.tar"
                save_checkpoint(chkpt_state, filename=chkpt_filename, directory=config.CHECKPOINT_DIR)

                torch.save(netG_A2B.module.state_dict(), os.path.join(config.CHECKPOINT_DIR, f'{epoch:03d}_netG_A2B.pth'))
                torch.save(netG_B2A.module.state_dict(), os.path.join(config.CHECKPOINT_DIR, f'{epoch:03d}_netG_B2A.pth'))
                torch.save(netG_A2B.module.state_dict(), os.path.join(config.CHECKPOINT_DIR, f'latest_netG_A2B.pth'))
                torch.save(netG_B2A.module.state_dict(), os.path.join(config.CHECKPOINT_DIR, f'latest_netG_B2A.pth'))

            if epoch % config.SAVE_SAMPLES_EPOCH_FREQ == 0:
                sample_batch = {'A': real_A, 'B': real_B}
                # Pass the underlying models (.module) to save_sample_images
                save_sample_images(epoch, sample_batch, netG_A2B.module, netG_B2A.module, device, max_samples=4)


    # --- End of Training ---
    print(f"[Rank {rank}] Training finished.")
    cleanup_ddp()


if __name__ == "__main__":
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f"Starting training script. World Size: {world_size}, Global Rank: {rank}, Local Rank: {local_rank}")

    if world_size > 1 and not dist.is_available():
         raise RuntimeError("Distributed training requested but Distributed module not available.")

    if torch.cuda.is_available():
        if world_size > torch.cuda.device_count():
             print(f"Warning: WORLD_SIZE ({world_size}) > available GPUs ({torch.cuda.device_count()}). Some processes might idle or share GPUs.")
        if local_rank >= torch.cuda.device_count():
             raise RuntimeError(f"LOCAL_RANK {local_rank} is invalid for available GPUs ({torch.cuda.device_count()})")
    else:
        if world_size > 1:
            raise RuntimeError("Distributed training requested but no CUDA GPUs available.")
        print("Warning: CUDA not available, running on CPU (DDP setup will be skipped).")

    main(rank, world_size)
