# src/utils.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.distributed as dist

import os
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import GradScaler only if AMP is intended to be used globally or conditionally
# This allows the utils file to be imported even if AMP is not used or CUDA not available
try:
    from torch.cuda.amp import GradScaler
except ImportError:
    # Define a dummy GradScaler if torch.cuda.amp is not available
    class GradScaler:
        def __init__(self, *args, **kwargs):
            self._enabled = kwargs.get('enabled', False) # Track if it *should* be enabled
            if self._enabled:
                 print("Warning: torch.cuda.amp.GradScaler not available. AMP disabled.")
            self._enabled = False # Force disable if import failed
        def scale(self, loss): return loss
        def step(self, optimizer): optimizer.step()
        def update(self): pass
        def load_state_dict(self, state_dict):
            # Only print warning if loading was attempted on an enabled dummy
             if self._enabled: print("Warning: Attempting to load state into dummy GradScaler.")
        def state_dict(self): return {}
        def get_scale(self): return 1.0
        def is_enabled(self): return self._enabled


from src import config # Import config for paths and parameters

# -------------------------
# DDP Helper Functions
# -------------------------

def setup_ddp(rank: int, world_size: int):
    """Initializes the distributed process group."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500') # Default port
    # Initialize the process group
    # backend='nccl' is recommended for NVIDIA GPUs
    backend = 'nccl' if torch.cuda.is_available() else 'gloo' # Use gloo for CPU-only fallback
    if not torch.cuda.is_available() and world_size > 1:
        print(f"Warning: CUDA not available, using 'gloo' backend for DDP on CPU.")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank) # Pin the process to a specific GPU
        print(f"[Rank {rank}] DDP Initialized (NCCL). Device: cuda:{rank}")
    else:
        print(f"[Rank {rank}] DDP Initialized (gloo) on CPU.")


def cleanup_ddp():
    """Cleans up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("DDP Cleaned up.")

def is_main_process():
    """Checks if the current process is the main process (rank 0)."""
    if not dist.is_available() or not dist.is_initialized():
        return True # Not using DDP or not initialized yet
    return dist.get_rank() == 0

# -------------------------
# Weight Initialization
# -------------------------

def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initializes network weights.
    :param net: Network to initialize
    :param init_type: Initialization method ('normal', 'xavier', 'kaiming', 'orthogonal')
    :param init_gain: Scaling factor for normal, xavier, and orthogonal.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'Initialization method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print(f'Initializing network with {init_type} strategy (gain={init_gain})')
    net.apply(init_func)

# -------------------------
# Learning Rate Scheduler
# -------------------------

def get_scheduler(optimizer, cfg):
    """
    Returns a learning rate scheduler that implements linear decay.
    :param optimizer: The optimizer linked to this scheduler
    :param cfg: Configuration object (using attributes like NUM_EPOCHS, START_EPOCH, EPOCH_DECAY_START)
    """
    # Correct calculation for total decay epochs
    total_epochs = cfg.NUM_EPOCHS
    start_epoch = cfg.START_EPOCH # The epoch number we start from (e.g., 1 or loaded from checkpoint)
    decay_start_epoch_abs = cfg.EPOCH_DECAY_START # The absolute epoch number when decay starts (e.g., 100)

    # Check if decay is needed at all based on total epochs vs decay start
    if decay_start_epoch_abs >= total_epochs:
        print("Learning rate decay is disabled as EPOCH_DECAY_START >= NUM_EPOCHS.")
        # Return a scheduler that does nothing (constant LR)
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    # Calculate the number of epochs for decay
    # Total decay duration = Total epochs - Decay start epoch + 1 (to include the last epoch)
    num_decay_epochs = total_epochs - decay_start_epoch_abs + 1
    if num_decay_epochs <= 0:
         print(f"Warning: Calculated num_decay_epochs is {num_decay_epochs}. Decay might not work as expected. Check NUM_EPOCHS and EPOCH_DECAY_START.")
         num_decay_epochs = 1 # Avoid division by zero

    def lambda_rule(epoch):
        """
        Calculate the learning rate multiplier.
        'epoch' parameter from LambdaLR is 0-based and counts training epochs completed *after* this scheduler starts.
        """
        # Calculate the *absolute* current epoch number during training
        # current_abs_epoch = start_epoch + epoch # 'epoch' is 0 for the first step after scheduler creation/resume
        # Let's use the epoch number directly passed by the training loop for clarity if possible,
        # but LambdaLR provides its own 'epoch' counter. Assume 'epoch' is relative to the scheduler's start.
        # We need to map LambdaLR's 'epoch' counter to the absolute training epoch number.
        # Let's assume the scheduler is stepped *after* epoch `e` completes. Then `scheduler.last_epoch` is `e`.
        # The input `epoch` to lambda_rule is `scheduler.last_epoch + 1`.
        # So, `current_abs_epoch = start_epoch + epoch`.
        # Let's rethink: LambdaLR applies the rule based on its internal counter `last_epoch`.
        # When resuming from epoch `s`, we set `scheduler.last_epoch = s - 1`.
        # For the first step (at the end of epoch `s`), lambda_rule gets `epoch = s`.
        # For the step at the end of epoch `s+1`, lambda_rule gets `epoch = s + 1`.
        # So, the input `epoch` to lambda_rule corresponds to the epoch number *just completed*.

        # Calculate how many epochs we are *into* the decay phase
        epochs_into_decay = max(0, epoch - (decay_start_epoch_abs - 1))

        # Calculate multiplier: 1.0 for epochs before decay, linear decay afterwards
        lr_l = 1.0 - epochs_into_decay / float(num_decay_epochs)

        # Ensure multiplier doesn't go below zero
        lr_l = max(0.0, lr_l)
        # print(f"Scheduler Debug: epoch={epoch}, decay_start_epoch_abs={decay_start_epoch_abs}, num_decay_epochs={num_decay_epochs}, epochs_into_decay={epochs_into_decay}, lr_l={lr_l}")
        return lr_l

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    print(f"Initialized linear decay scheduler: starts decaying after epoch {decay_start_epoch_abs - 1}, ends at epoch {total_epochs}.")
    return scheduler


# -------------------------
# Image Pool (for Discriminator Stability)
# -------------------------

class ImagePool():
    """
    This class implements an image buffer that stores previously generated images.
    This buffer allows us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators, stabilizing training.
    """
    def __init__(self, pool_size):
        """
        Initialize the ImagePool class
        :param pool_size: the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """
        Return images from the pool.
        :param images: the latest generated images from the generator (Tensor B, C, H, W)
        :return: images from the buffer (Tensor B, C, H, W).

        By 50% chance, the buffer will return input images.
        By 50% chance, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images: # Iterate through images in the batch
            image = torch.unsqueeze(image.data, 0) # Add batch dimension back: (C, H, W) -> (1, C, H, W)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; store current image
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive [0, pool_size-1]
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        # Collect all the images and stack them back into a single batch tensor
        return_images = torch.cat(return_images, 0) # Shape: (B, C, H, W)
        return return_images

# -------------------------
# Tensor to Image Conversion
# -------------------------

def tensor2im(input_image, imtype=np.uint8, scale_to_0_1=False):
    """
    Converts a Tensor array into a numpy image array.
    Assumes the tensor is in the format [C, H, W] or [B, C, H, W].
    Handles denormalization from [-1, 1] range.

    :param input_image: the input image tensor array
    :param imtype: the desired type of the converted numpy array (e.g., np.uint8)
    :param scale_to_0_1: If true, scales to [0, 1] float instead of [0, 255] uint8.
    :return: A numpy array representing the image(s). Shape [B, H, W], [H, W], [B, H, W, C], or [H, W, C]
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image # Return as is if not tensor or numpy

        # Move tensor to CPU and convert to numpy
        image_numpy = image_tensor.cpu().float().numpy()

        # Denormalize from [-1, 1] to [0, 1]
        image_numpy = (image_numpy + 1) / 2.0

        # Handle potential batch dimension and transpose C,H,W -> H,W,C or B,C,H,W -> B,H,W,C
        if image_numpy.ndim == 4: # B, C, H, W
            image_numpy = np.transpose(image_numpy, (0, 2, 3, 1)) # B, H, W, C
        elif image_numpy.ndim == 3: # C, H, W
            image_numpy = np.transpose(image_numpy, (1, 2, 0)) # H, W, C
        elif image_numpy.ndim == 2: # H, W (e.g., if input was already H, W) - unlikely if normalized
            pass # Keep as H, W
        else: # Unexpected dimension
             print(f"Warning: tensor2im received unexpected tensor dimension: {image_numpy.ndim}. Returning raw numpy array.")
             return image_numpy

        # Handle single channel (grayscale) - remove the channel dim for saving/display
        # Check the *last* dimension after transpose
        if image_numpy.ndim == 3 and image_numpy.shape[-1] == 1: # H, W, 1
             image_numpy = image_numpy.squeeze(axis=-1) # H, W
        elif image_numpy.ndim == 4 and image_numpy.shape[-1] == 1: # B, H, W, 1
             image_numpy = image_numpy.squeeze(axis=-1) # B, H, W

        # Scale to target range
        if scale_to_0_1:
            image_numpy = np.clip(image_numpy, 0, 1) # Clip just in case
        else:
            image_numpy = np.clip(image_numpy * 255.0, 0, 255) # Scale to [0, 255] and clip

    else:  # if it is already a numpy array, maybe just clip and typecast?
        image_numpy = input_image
        if scale_to_0_1:
            image_numpy = np.clip(image_numpy, 0, 1)
        else:
             image_numpy = np.clip(image_numpy, 0, 255)


    # Convert to the target type if not scaling to float [0,1]
    if not scale_to_0_1:
        return image_numpy.astype(imtype)
    else:
        # Ensure float type if scaling to [0, 1]
        return image_numpy.astype(np.float32) if image_numpy.dtype != np.float32 else image_numpy

# -------------------------
# Image Saving
# -------------------------

def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """
    Saves a single numpy image to the specified path using matplotlib.
    Input image_numpy should be in H, W (grayscale) or H, W, C (color) format.

    :param image_numpy: Single image array to save (2D or 3D).
    :param image_path: Path where the image will be saved.
    :param aspect_ratio: Aspect ratio for display (used by imshow if needed, not by imsave).
    """
    # Check if grayscale (H, W) or color (H, W, C)
    if image_numpy.ndim == 2: # Grayscale H, W
        plt.imsave(image_path, image_numpy, cmap='gray')
    elif image_numpy.ndim == 3: # Color H, W, C or Grayscale H, W, 1
         if image_numpy.shape[2] == 1: # Grayscale (H, W, 1) -> Squeeze to H, W
              plt.imsave(image_path, image_numpy.squeeze(axis=2), cmap='gray')
         elif image_numpy.shape[2] in [3, 4]: # Color (H, W, 3/4)
              # Clip values to [0, 255] if uint8, or [0, 1] if float before saving
              if image_numpy.dtype == np.uint8:
                  image_numpy = np.clip(image_numpy, 0, 255)
              elif image_numpy.dtype == np.float32 or image_numpy.dtype == np.float64:
                   image_numpy = np.clip(image_numpy, 0, 1) # Assuming float images are [0, 1] for imsave
              plt.imsave(image_path, image_numpy)
         else:
              print(f"Warning: Could not save image. Unexpected channel size in 3D array: {image_numpy.shape}")
    else:
        print(f"Warning: Could not save image. Unexpected numpy array dimension: {image_numpy.ndim} with shape {image_numpy.shape}")


def save_sample_images(epoch, batch, netG_A2B, netG_B2A, device, max_samples=4):
    """
    Saves a grid of sample images during training (Real A, Fake B, Rec A, Real B, Fake A, Rec B).
    Only runs on the main process in DDP. Assumes input batch is already on the correct device.
    """
    if not is_main_process():
        return

    # Ensure models are in eval mode for inference, but keep track of original mode
    was_training_G_A2B = netG_A2B.training
    was_training_G_B2A = netG_B2A.training
    netG_A2B.eval()
    netG_B2A.eval()

    # Get data (assumes batch is a dictionary {'A': tensor, 'B': tensor})
    real_A = batch['A'].to(device) # Ensure data is on the correct device
    real_B = batch['B'].to(device)

    with torch.no_grad():
        # Limit number of samples to display/save
        num_samples = min(real_A.size(0), max_samples)
        if num_samples == 0:
             print("Warning: No samples available in the batch to save.")
             return

        real_A = real_A[:num_samples]
        real_B = real_B[:num_samples]

        # Generate images
        fake_B = netG_A2B(real_A)
        rec_A = netG_B2A(fake_B)
        fake_A = netG_B2A(real_B)
        rec_B = netG_A2B(fake_A)

    # Restore original training mode
    if was_training_G_A2B: netG_A2B.train()
    if was_training_G_B2A: netG_B2A.train()

    # Convert tensors to displayable numpy images (uint8 [0, 255])
    # tensor2im returns shape (num_samples, H, W) for grayscale batches
    real_A_np = tensor2im(real_A)
    fake_B_np = tensor2im(fake_B)
    rec_A_np = tensor2im(rec_A)
    real_B_np = tensor2im(real_B)
    fake_A_np = tensor2im(fake_A)
    rec_B_np = tensor2im(rec_B)

    # Determine grid size
    cols = 6 # RealA, FakeB, RecA, RealB, FakeA, RecB
    rows = num_samples

    # Check if conversion resulted in numpy arrays
    if not isinstance(real_A_np, np.ndarray):
         print(f"Warning: tensor2im did not return a numpy array for real_A. Skipping sample saving.")
         return
    if real_A_np.ndim != 3 or real_A_np.shape[0] != rows: # Expecting (rows, H, W)
         print(f"Warning: Unexpected numpy array shape after tensor2im: {real_A_np.shape}. Expected ({rows}, H, W). Skipping sample saving.")
         # This might happen if batch size was 0 or tensor2im failed
         return

    fig = plt.figure(figsize=(cols * 2.5, rows * 2.5)) # Adjust size as needed
    gs = gridspec.GridSpec(rows, cols, wspace=0.05, hspace=0.05)

    img_sets = [real_A_np, fake_B_np, rec_A_np, real_B_np, fake_A_np, rec_B_np]
    titles = ['Real A (LQ)', 'Fake B (HQ)', 'Rec A (LQ)', 'Real B (HQ)', 'Fake A (LQ)', 'Rec B (HQ)']

    for i in range(rows): # Iterate through samples in the batch
        for j in range(cols): # Iterate through image types (Real A, Fake B, etc.)
            ax = plt.subplot(gs[i, j])
            plt.axis('off')
            ax.set_aspect('equal')
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            # --- Correction Here ---
            # Select the i-th image from the j-th set of images
            # img_sets[j] has shape (rows, H, W), so img_sets[j][i] has shape (H, W)
            img_to_show = img_sets[j][i]
            # -----------------------

            # Add title only to the first row
            if i == 0:
                 ax.set_title(titles[j], fontsize=10)

            # Display grayscale image using 'gray' colormap
            plt.imshow(img_to_show, cmap='gray')

    # Save the figure
    save_filename = f'epoch_{epoch:03d}_samples.png'
    save_path = os.path.join(config.TRAIN_SAMPLES_DIR, save_filename)
    try:
        plt.savefig(save_path, bbox_inches='tight', dpi=150) # Increase dpi for better resolution
    except Exception as e:
        print(f"Error saving sample image grid: {e}")
    finally:
        plt.close(fig) # Close the figure to free memory
    # print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Saved training sample: {save_path}")


# -------------------------
# Checkpointing Functions
# -------------------------

def save_checkpoint(state, filename="my_checkpoint.pth.tar", directory=config.CHECKPOINT_DIR, is_best=False, best_filename="best_model.pth.tar"):
    """
    Saves model checkpoint. Includes handling for DDP models and expects AMP scaler states in `state`.
    Only saves on the main process.

    :param state: Dictionary containing model and optimizer states, epoch, scaler states, etc.
                  Example: {'epoch': epoch,
                            'netG_A2B_state_dict': model_G_A2B.module.state_dict(), # Use .module with DDP
                            'netG_B2A_state_dict': model_G_B2A.module.state_dict(),
                            'netD_A_state_dict': model_D_A.module.state_dict(),
                            'netD_B_state_dict': model_D_B.module.state_dict(),
                            'optimizer_G_state_dict': optimizer_G.state_dict(),
                            'optimizer_D_state_dict': optimizer_D.state_dict(),
                            'scheduler_G_state_dict': scheduler_G.state_dict(),
                            'scheduler_D_state_dict': scheduler_D.state_dict(),
                            'scaler_G_state_dict': scaler_G.state_dict(), # <<< Added
                            'scaler_D_state_dict': scaler_D.state_dict()  # <<< Added
                           }
    :param filename: Filename for the checkpoint.
    :param directory: Directory to save the checkpoint.
    :param is_best: If true, also saves a copy as best_filename. (Not implemented fully here)
    :param best_filename: Filename for the best model checkpoint.
    """
    if not is_main_process():
        return # Only save on rank 0

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    filepath = os.path.join(directory, filename)
    try:
        torch.save(state, filepath)
        print(f"==> Saved checkpoint '{filepath}' (Epoch {state.get('epoch', '?')})")

        if is_best: # Simple 'best' saving - copies the file
            best_filepath = os.path.join(directory, best_filename)
            import shutil
            shutil.copyfile(filepath, best_filepath)
            print(f"==> Saved new best model '{best_filepath}'")
    except Exception as e:
        print(f"Error saving checkpoint {filepath}: {e}")



def load_checkpoint(checkpoint_path, netG_A2B, netG_B2A, netD_A, netD_B,
                    optimizer_G=None, optimizer_D=None, scheduler_G=None, scheduler_D=None,
                    device='cuda', scaler_G=None, scaler_D=None):
    """
    Loads model checkpoint. Handles DDP model state dict keys and AMP scalers.

    :param checkpoint_path: Path to the checkpoint file.
    :param netG_A2B, netG_B2A, netD_A, netD_B: Model instances to load weights into.
    :param optimizer_G, optimizer_D: Optimizer instances to load states into (optional).
    :param scheduler_G, scheduler_D: Scheduler instances to load states into (optional).
    :param device: Device to load the checkpoint onto ('cuda' or 'cpu') or target device for optimizer states.
    :param scaler_G: GradScaler instance for generator (optional, needed if training with AMP).
    :param scaler_D: GradScaler instance for discriminator (optional, needed if training with AMP).
    :return: The epoch number from the checkpoint (integer).
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"==> Loading checkpoint '{checkpoint_path}'")
    # Load checkpoint onto CPU memory first to avoid GPU OOM, especially with DDP.
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    start_epoch = checkpoint.get('epoch', 1) # Default to 1 if epoch key not found

    # Helper function to load state dict, handling 'module.' prefix from DDP
    def load_state_dict_helper(model, state_dict):
        # Create new state_dict with 'module.' prefix removed if necessary
        new_state_dict = {}
        # Check if the checkpoint state_dict has the 'module.' prefix (saved from DDP)
        has_module_prefix = any(key.startswith('module.') for key in state_dict)
        # Check if the current model instance is wrapped in DDP
        is_ddp_model = isinstance(model, nn.parallel.DistributedDataParallel)

        if is_ddp_model and not has_module_prefix:
             # Loading non-DDP checkpoint into current DDP model
             print("Loading non-DDP checkpoint into DDP model. Applying to model.module.")
             model.module.load_state_dict(state_dict, strict=False)
        elif not is_ddp_model and has_module_prefix:
            # Loading DDP checkpoint into current non-DDP model
            print("Loading DDP checkpoint into non-DDP model. Removing 'module.' prefix.")
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v # remove `module.`
                else:
                    new_state_dict[k] = v # Keep non-module keys if any
            model.load_state_dict(new_state_dict, strict=False)
        elif is_ddp_model and has_module_prefix:
             # Loading DDP checkpoint into current DDP model
             print("Loading DDP checkpoint into DDP model. Applying to model.module.")
             # DDP wrapper expects state_dict with 'module.' prefix when loading directly onto the wrapped model
             # However, it's often safer to load onto model.module directly
             try:
                 # Try loading directly onto the underlying module
                 model.module.load_state_dict(state_dict, strict=False)
             except RuntimeError as e1:
                 print(f"Loading onto model.module failed: {e1}. Trying load_state_dict on DDP wrapper.")
                 # Fallback: Try loading onto the DDP wrapper itself (might require matching 'module.' prefix)
                 try:
                      model.load_state_dict(state_dict, strict=False)
                 except RuntimeError as e2:
                      print(f"Loading onto DDP wrapper also failed: {e2}.")
                      # As a last resort, try loading without the prefix onto the module
                      print("Last resort: removing 'module.' prefix and loading onto model.module.")
                      new_state_dict = {}
                      for k, v in state_dict.items():
                           if k.startswith('module.'):
                               new_state_dict[k[7:]] = v
                           else:
                               new_state_dict[k] = v
                      model.module.load_state_dict(new_state_dict, strict=False)

        elif not is_ddp_model and not has_module_prefix:
             # Loading non-DDP checkpoint into non-DDP model
             print("Loading non-DDP checkpoint into non-DDP model.")
             model.load_state_dict(state_dict, strict=False)
        else: # Fallback / Unexpected case
             print(f"Warning: State dict loading condition unhandled (is_ddp={is_ddp_model}, has_prefix={has_module_prefix}). Trying direct load with strict=False.")
             try:
                  model.load_state_dict(state_dict, strict=False)
             except RuntimeError as e:
                  print(f"Direct load failed: {e}. Attempting load onto underlying module if DDP.")
                  if is_ddp_model:
                       model.module.load_state_dict(state_dict, strict=False)


    # Load models
    if 'netG_A2B_state_dict' in checkpoint:
        load_state_dict_helper(netG_A2B, checkpoint['netG_A2B_state_dict'])
        print("Loaded Generator A->B weights.")
    else: print("Warning: netG_A2B_state_dict not found in checkpoint.")

    if 'netG_B2A_state_dict' in checkpoint:
        load_state_dict_helper(netG_B2A, checkpoint['netG_B2A_state_dict'])
        print("Loaded Generator B->A weights.")
    else: print("Warning: netG_B2A_state_dict not found in checkpoint.")

    if 'netD_A_state_dict' in checkpoint:
        load_state_dict_helper(netD_A, checkpoint['netD_A_state_dict'])
        print("Loaded Discriminator A weights.")
    else: print("Warning: netD_A_state_dict not found in checkpoint.")

    if 'netD_B_state_dict' in checkpoint:
        load_state_dict_helper(netD_B, checkpoint['netD_B_state_dict'])
        print("Loaded Discriminator B weights.")
    else: print("Warning: netD_B_state_dict not found in checkpoint.")


    # Load optimizers if provided and available in checkpoint
    if optimizer_G and 'optimizer_G_state_dict' in checkpoint:
        try:
            optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            print("Loaded Optimizer G state.")
            # Move optimizer state tensors to the correct device
            # Important after loading to CPU with map_location='cpu'
            for state in optimizer_G.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            print(f"Moved Optimizer G state to device: {device}")
        except Exception as e:
             print(f"Warning: Could not load Optimizer G state properly. Error: {e}. Optimizer might be re-initialized.")

    if optimizer_D and 'optimizer_D_state_dict' in checkpoint:
         try:
            optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            print("Loaded Optimizer D state.")
            # Move optimizer state tensors to the correct device
            for state in optimizer_D.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            print(f"Moved Optimizer D state to device: {device}")
         except Exception as e:
             print(f"Warning: Could not load Optimizer D state properly. Error: {e}. Optimizer might be re-initialized.")


    # Load schedulers if provided and available
    # Schedulers usually don't have tensors requiring device transfer, but load them after optimizers
    if scheduler_G and 'scheduler_G_state_dict' in checkpoint:
        try:
            scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
            print("Loaded Scheduler G state.")
        except Exception as e:
            print(f"Warning: Could not load Scheduler G state. Error: {e}")

    if scheduler_D and 'scheduler_D_state_dict' in checkpoint:
        try:
            scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
            print("Loaded Scheduler D state.")
        except Exception as e:
            print(f"Warning: Could not load Scheduler D state. Error: {e}")


    # --- AMP: Load scaler states if provided and available in checkpoint ---
    # Ensure scaler objects exist before trying to load state into them
    if scaler_G is not None and 'scaler_G_state_dict' in checkpoint:
        if isinstance(scaler_G, GradScaler): # Check if it's a real GradScaler
            try:
                scaler_G.load_state_dict(checkpoint['scaler_G_state_dict'])
                print("Loaded Scaler G state.")
            except Exception as e:
                 print(f"Warning: Could not load Scaler G state. Error: {e}. Scaler might use default state.")
        else:
             print("Warning: scaler_G provided but is not a GradScaler instance. Skipping state load.")

    if scaler_D is not None and 'scaler_D_state_dict' in checkpoint:
        if isinstance(scaler_D, GradScaler): # Check if it's a real GradScaler
            try:
                scaler_D.load_state_dict(checkpoint['scaler_D_state_dict'])
                print("Loaded Scaler D state.")
            except Exception as e:
                 print(f"Warning: Could not load Scaler D state. Error: {e}. Scaler might use default state.")
        else:
            print("Warning: scaler_D provided but is not a GradScaler instance. Skipping state load.")
    # -----------------------------------------------------------------------


    print(f"==> Checkpoint loaded successfully. Resuming from epoch {start_epoch}")
    # Return the epoch number to resume from (checkpoint saves the *next* epoch to run)
    return start_epoch
