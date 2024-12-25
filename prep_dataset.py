import torch
from torchvision import datasets, transforms
import numpy as np
import ssl

# Bypass SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Define patch size
PATCH_SIZE = 16

# Step 1: Download and Load the CIFAR-10 Dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Resize((32, 32)),  # Ensure all images are the same size (CIFAR-10 is already 32x32)
])

dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# Step 2: Preprocess to Create Patches
def create_patches(images, patch_size):
    """
    Split the batch of images into patches.

    Args:
        images (torch.Tensor): Batch of images with shape (B, C, H, W).
        patch_size (int): Size of each square patch.

    Returns:
        torch.Tensor: Flattened patches with shape (B, N, P*P*C),
                      where N is the number of patches per image.
    """
    B, C, H, W = images.shape
    assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions must be divisible by patch size."

    # Number of patches along each dimension
    patches_per_dim = H // patch_size
    
    # Reshape and flatten
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5)  # Rearrange to (B, patches_y, patches_x, C, P, P)
    patches = patches.reshape(B, -1, patch_size * patch_size * C)  # Flatten to (B, N, P*P*C)

    return patches

# Step 3: Iterate through the DataLoader and Create Patches
for batch in loader:
    images, labels = batch
    patches = create_patches(images, PATCH_SIZE)
    print(f"Original Image Shape: {images.shape}")
    print(f"Patches Shape: {patches.shape}")
    break
