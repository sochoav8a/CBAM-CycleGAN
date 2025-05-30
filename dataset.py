# src/dataset.py

import torch
import torch.utils.data as data
import numpy as np
import torchvision.transforms as transforms
import os
import random
from PIL import Image # Aunque no usemos PIL directamente, es útil para entender el flujo de ToTensor

class NpzImageDataset(data.Dataset):
    """
    Dataset class for loading unpaired (train) or paired (val/test) images
    from NumPy NPZ files.

    Assumes NPZ file contains 'arr_0' (domain A, e.g., LQ) and 'arr_1' (domain B, e.g., HQ).
    Images are assumed to be grayscale [H, W], float32, range [0.0, 255.0].
    """

    def __init__(self, npz_path, mode='train'):
        """
        Initialize the dataset loader.

        Args:
            npz_path (str): Path to the .npz file.
            mode (str): Operating mode: 'train' for unpaired loading,
                        'val' or 'test' for paired loading.
        """
        super(NpzImageDataset, self).__init__()

        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ file not found at: {npz_path}")

        print(f"Loading data from: {npz_path} (Mode: {mode})")
        try:
            # Load the entire NPZ file into memory
            npz_data = np.load(npz_path)
        except Exception as e:
            raise IOError(f"Error loading NPZ file {npz_path}: {e}")

        # Check if expected arrays are present
        if 'arr_0' not in npz_data:
            raise KeyError("Key 'arr_0' (domain A / LQ images) not found in NPZ file.")
        if 'arr_1' not in npz_data:
            raise KeyError("Key 'arr_1' (domain B / HQ images) not found in NPZ file.")

        self.data_A = npz_data['arr_0'] # Domain A (e.g., LQ)
        self.data_B = npz_data['arr_1'] # Domain B (e.g., HQ)

        # Validate data properties (optional but recommended)
        if self.data_A.dtype != np.float32:
             print(f"Warning: Data A dtype is {self.data_A.dtype}, expected float32. Converting...")
             self.data_A = self.data_A.astype(np.float32)
        if self.data_B.dtype != np.float32:
             print(f"Warning: Data B dtype is {self.data_B.dtype}, expected float32. Converting...")
             self.data_B = self.data_B.astype(np.float32)

        # Basic shape check (assuming N, H, W format for grayscale)
        if self.data_A.ndim != 3 or self.data_B.ndim != 3:
             raise ValueError(f"Expected data shape (N, H, W), but got A: {self.data_A.shape}, B: {self.data_B.shape}")
        if self.data_A.shape[1:] != self.data_B.shape[1:] and mode != 'train':
             # Allow different spatial sizes only in train mode if necessary, but warn
             print(f"Warning: Spatial dimensions differ between A {self.data_A.shape[1:]} and B {self.data_B.shape[1:]}.")


        self.len_A = len(self.data_A)
        self.len_B = len(self.data_B)
        self.mode = mode

        if self.mode != 'train': # Validation or Test mode (paired)
            if self.len_A != self.len_B:
                # In paired mode, lengths must match. We might choose to truncate or raise an error.
                # Raising an error is safer for validation/testing.
                raise ValueError(f"In '{self.mode}' mode, datasets A and B must have the same number of images for pairing. "
                                 f"Got len(A)={self.len_A}, len(B)={self.len_B}")
            self.dataset_len = self.len_A # or self.len_B
            print(f"Initialized paired dataset (mode='{mode}') with {self.dataset_len} image pairs.")
        else: # Train mode (unpaired)
            # Length of dataset is the maximum of the two domains
            self.dataset_len = max(self.len_A, self.len_B)
            print(f"Initialized unpaired dataset (mode='train') with len(A)={self.len_A}, len(B)={self.len_B}. Effective length: {self.dataset_len}")


        # Define transformations: Convert numpy [H, W] -> tensor [C, H, W] and normalize [-1, 1]
        # Input numpy array is assumed: float32, H, W, range [0, 255]
        self.transform = transforms.Compose([
            # 1. Convert numpy [H, W] float32 -> tensor [H, W] float32
            transforms.Lambda(lambda x: torch.from_numpy(x.astype(np.float32))),
            # 2. Add channel dimension: [H, W] -> [1, H, W] (for grayscale)
            transforms.Lambda(lambda x: x.unsqueeze(0)),
            # 3. Normalize from [0, 255] to [-1, 1]
            transforms.Lambda(lambda x: (x / 127.5) - 1.0)
        ])

        # Alternative using standard transforms (requires numpy uint8 HWC or PIL input to ToTensor)
        # If input were uint8 HWC [0, 255] numpy:
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(), # Converts numpy HWC [0, 255] uint8 -> Tensor C, H, W [0, 1] float
        #     transforms.Normalize(mean=[0.5], std=[0.5]) # Maps [0, 1] -> [-1, 1] (adjust mean/std for #channels)
        # ])


    def __getitem__(self, index):
        """
        Return a data sample.
        - In 'train' mode: returns a random image from A and a random image from B.
        - In 'val'/'test' mode: returns the paired images at the given index.
        """
        if self.mode == 'train':
            # Unpaired: Sample randomly from each domain
            # Use modulo to handle potential index wrapping if dataset_len > len_A or len_B
            index_A = random.randint(0, self.len_A - 1)
            index_B = random.randint(0, self.len_B - 1)
            # Or use the input index for one domain and random for the other, ensures iteration over longer dataset
            # index_A = index % self.len_A
            # index_B = random.randint(0, self.len_B - 1)

            item_A = self.data_A[index_A]
            item_B = self.data_B[index_B]
        else:
            # Paired: Use the same index for both domains
            # Index validation already happened in __init__ and via __len__
            if index >= self.dataset_len:
                 raise IndexError(f"Index {index} out of bounds for dataset length {self.dataset_len}")
            item_A = self.data_A[index]
            item_B = self.data_B[index]

        # Apply transformations
        img_A = self.transform(item_A)
        img_B = self.transform(item_B)

        return {'A': img_A, 'B': img_B, 'A_path': f'A_{index_A if self.mode == "train" else index}', 'B_path': f'B_{index_B if self.mode == "train" else index}'} # Add paths/ids if needed for debugging


    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.dataset_len


# --- Example Usage Block (for testing the dataset class) ---
if __name__ == '__main__':
    print("\n--- Testing NpzImageDataset ---")

    # Create dummy NPZ files for testing
    dummy_train_path = "dummy_train_data.npz"
    dummy_val_path = "dummy_val_data.npz"
    H, W = 64, 64 # Smaller size for faster testing

    # Create dummy training data (unpaired)
    train_A = np.random.rand(10, H, W).astype(np.float32) * 255.0 # 10 images LQ
    train_B = np.random.rand(12, H, W).astype(np.float32) * 255.0 # 12 images HQ
    np.savez(dummy_train_path, arr_0=train_A, arr_1=train_B)
    print(f"Created dummy train file: {dummy_train_path} (A:{train_A.shape}, B:{train_B.shape})")

    # Create dummy validation data (paired)
    val_A = np.random.rand(5, H, W).astype(np.float32) * 255.0 # 5 images LQ
    val_B = val_A * 0.8 + 50 # Create pseudo-paired HQ by simple transform
    val_B = np.clip(val_B, 0, 255).astype(np.float32)
    np.savez(dummy_val_path, arr_0=val_A, arr_1=val_B)
    print(f"Created dummy validation file: {dummy_val_path} (A:{val_A.shape}, B:{val_B.shape})")

    # --- Test Train Mode (Unpaired) ---
    print("\nTesting Train Mode (Unpaired)...")
    try:
        train_dataset = NpzImageDataset(npz_path=dummy_train_path, mode='train')
        print(f"Train dataset length: {len(train_dataset)}")
        assert len(train_dataset) == 12 # Should be max(10, 12)

        # Get a sample
        sample_train = train_dataset[0]
        img_A_train = sample_train['A']
        img_B_train = sample_train['B']

        print(f"Sample A shape: {img_A_train.shape}, dtype: {img_A_train.dtype}, min: {img_A_train.min():.2f}, max: {img_A_train.max():.2f}")
        print(f"Sample B shape: {img_B_train.shape}, dtype: {img_B_train.dtype}, min: {img_B_train.min():.2f}, max: {img_B_train.max():.2f}")

        # Check shape, type and normalization range
        assert img_A_train.shape == (1, H, W)
        assert img_B_train.shape == (1, H, W)
        assert img_A_train.dtype == torch.float32
        assert img_B_train.dtype == torch.float32
        assert -1.01 <= img_A_train.min() <= -0.9 and 0.9 <= img_A_train.max() <= 1.01 # Check range [-1, 1] approx
        assert -1.01 <= img_B_train.min() <= -0.9 and 0.9 <= img_B_train.max() <= 1.01

        # Check if different images are returned (highly likely with random sampling)
        sample1 = train_dataset[1]
        sample2 = train_dataset[2]
        # Note: Comparing random tensors might yield false negatives if indices collide by chance.
        # A better check would be to compare paths/indices if they were stored, but basic check:
        print("Train mode test completed.")

    except Exception as e:
        print(f"Error during train mode test: {e}")
        raise e


    # --- Test Validation Mode (Paired) ---
    print("\nTesting Validation Mode (Paired)...")
    try:
        val_dataset = NpzImageDataset(npz_path=dummy_val_path, mode='val')
        print(f"Validation dataset length: {len(val_dataset)}")
        assert len(val_dataset) == 5 # Should be length of paired data

        # Get a sample
        sample_val = val_dataset[0]
        img_A_val = sample_val['A']
        img_B_val = sample_val['B']

        print(f"Sample A shape: {img_A_val.shape}, dtype: {img_A_val.dtype}, min: {img_A_val.min():.2f}, max: {img_A_val.max():.2f}")
        print(f"Sample B shape: {img_B_val.shape}, dtype: {img_B_val.dtype}, min: {img_B_val.min():.2f}, max: {img_B_val.max():.2f}")

        # Check shape, type and normalization range
        assert img_A_val.shape == (1, H, W)
        assert img_B_val.shape == (1, H, W)
        assert img_A_val.dtype == torch.float32
        assert img_B_val.dtype == torch.float32
        assert -1.01 <= img_A_val.min() <= 1.01 # Check range [-1, 1] approx
        assert -1.01 <= img_B_val.min() <= 1.01

        # Check pairing: Reconstruct original pixel value from tensor
        # Original val_A[0, 0, 0] = (tensor_A[0, 0, 0] + 1) * 127.5
        original_A_pixel_0_0 = (img_A_val[0, 0, 0].item() + 1.0) * 127.5
        original_B_pixel_0_0 = (img_B_val[0, 0, 0].item() + 1.0) * 127.5

        expected_B_pixel_0_0 = np.clip(val_A[0, 0, 0] * 0.8 + 50, 0, 255)

        # Compare with reasonable tolerance due to float precision
        # print(f"Original A[0,0,0]: {val_A[0,0,0]:.2f} -> Tensor A[0,0,0]: {img_A_val[0,0,0]:.2f} -> Recalc: {original_A_pixel_0_0:.2f}")
        # print(f"Original B[0,0,0]: {val_B[0,0,0]:.2f} -> Tensor B[0,0,0]: {img_B_val[0,0,0]:.2f} -> Recalc: {original_B_pixel_0_0:.2f}")
        # print(f"Expected B[0,0,0] based on A: {expected_B_pixel_0_0:.2f}")

        assert abs(original_A_pixel_0_0 - val_A[0, 0, 0]) < 1e-3, "Pairing check failed for A"
        assert abs(original_B_pixel_0_0 - val_B[0, 0, 0]) < 1e-3, "Pairing check failed for B"
        assert abs(original_B_pixel_0_0 - expected_B_pixel_0_0) < 1e-3, "Pairing check failed for B relative to A"


        print("Validation mode test completed.")

    except Exception as e:
        print(f"Error during validation mode test: {e}")
        raise e

    # Clean up dummy files
    print("\nCleaning up dummy files...")
    os.remove(dummy_train_path)
    os.remove(dummy_val_path)
    print("Dummy files removed.")

    print("\nNpzImageDataset test suite finished.")
