import numpy as np
import pickle
import matplotlib.pyplot as plt
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch

def load_data(split="train"):
    """
    A function to load the data.

    Parameters:
    - split: str, "train" or "test"

    Returns:
    - masked_images: np.ndarray, shape (n_images, channels, height, width), the images with random masks applied
    - masks: np.ndarray, shape (n_images, 4), the boundaries of the masks (left, bottom, right, top)
    - parametric: np.ndarray, shape (n_images, 3), the parametric representation of the images
    - description: list, the description of the images
    - images: np.ndarray, shape (n_images, channels, height, width), the original images
    """
    masked_images = []
    if split == "train":
        for i in range(5):
            masked_im_slice = np.load(f"data/masked_train_{i}.npy")
            masked_images.append(masked_im_slice)
        masked_images = np.concatenate(masked_images, axis=0)

        images = []
        for i in range(5):
            im_slice = np.load(f"data/images_train_{i}.npy")
            images.append(im_slice)
        images = np.concatenate(images, axis=0)
    else:
        masked_images = np.load(f"data/masked_test.npy")
        images = None

    description = pickle.load(open(f"data/desc_{split}.pkl", "rb"))

    parametric = np.load(f"data/param_{split}.npy")

    masks = np.load(f"data/mask_{split}.npy")


    return masked_images, masks, parametric, description, images


def compute_residual_images(img_set1, img_set2):
    #convert to float   
    img_set1 = img_set1.astype(np.float32)
    img_set2 = img_set2.astype(np.float32)
    # Ensure the shapes match
    assert img_set1.shape == img_set2.shape, "Image sets must have the same dimensions"
    
    # Compute the residuals
    residuals = np.abs(img_set1 - img_set2)
    return residuals

def calc_ssim_batched(img_set1, img_set2, batch_size=128):
    # Initialize the SSIM metric
    ssim = StructuralSimilarityIndexMeasure(data_range=255, reduction="none")
    
    # Ensure input tensors are of float32 type
    img_set1 = torch.tensor(img_set1, dtype=torch.float32)
    img_set2 = torch.tensor(img_set2, dtype=torch.float32)

    # Calculate SSIM in batches
    ssim_vals = []
    for i in range(0, len(img_set1), batch_size):
        batch_img1 = img_set1[i:i + batch_size]
        batch_img2 = img_set2[i:i + batch_size]
        
        # Compute SSIM for the batch and store the result
        ssim_batch = ssim(batch_img1, batch_img2)
        ssim_vals.append(ssim_batch.cpu().numpy())
    # Concatenate all batch results into a single array
    return np.concatenate(ssim_vals)
def calculate_composite_score(img_set1, img_set2, alpha=10, batch_size=128):
    ssim = calc_ssim_batched(img_set1, img_set2, batch_size=batch_size)
    ssim_rescaled = (ssim + 1) / 2
    residuals = compute_residual_images(img_set1, img_set2)
    MAE = residuals/255
    MAE = np.mean(MAE, axis=(1, 2, 3))
    composite_score = (ssim_rescaled + alpha*(1-MAE))/(1+alpha)
    return np.mean(composite_score), composite_score

# Function to visualize original images and their residuals
def display_images_residual(img_set1, img_set2):
    residuals = compute_residual_images(img_set1, img_set2)
    errors = np.mean(residuals, axis=(1, 2, 3))/255
    residuals = (255-residuals)/255    

    num_images = len(img_set1)

    plt.figure(figsize=(10, num_images * 3))
    
    for i in range(num_images):
        # Transpose the images to (height, width, channels)
        img1 = np.transpose(img_set1[i], (1, 2, 0))
        img2 = np.transpose(img_set2[i], (1, 2, 0))
        residual = np.transpose(residuals[i], (1, 2, 0))
        
        # Display images from the first set
        plt.subplot(num_images, 3, i * 3 + 1)
        plt.imshow(img1)
        plt.title("Image Set 1")
        plt.axis("off")
        
        # Display images from the second set
        plt.subplot(num_images, 3, i * 3 + 2)
        plt.imshow(img2)
        plt.title("Image Set 2")
        plt.axis("off")
        
        # Display residual images
        plt.subplot(num_images, 3, i * 3 + 3)
        plt.imshow(residual)
        plt.title(f"MAE: {errors[i]:.4f}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()

def apply_random_mask(images, minimum_proportion=0.4, maximum_proportion=0.8):
    """
    Apply a random rectangular mask to a batch of images and return the coordinates of the masked area.

    Parameters:
    - images: np.ndarray, shape (batch_size, channels, height, width)
    - minimum_proportion: float, minimum proportion of image dimensions for mask size
    - maximum_proportion: float, maximum proportion of image dimensions for mask size

    Returns:
    - masked_images: np.ndarray, images with applied random masks
    - mask_coordinates: np.ndarray, shape (batch_size, 4), coordinates (left, bottom, right, top) of masked area for each image
    """
    batch_size, channels, height, width = images.shape
    masked_images = images.copy()  # Copy images to avoid modifying the original array
    mask_coordinates = np.zeros((batch_size, 4), dtype=int)  # Array to store coordinates for each mask
    
    # Random mask sizes within the specified proportions
    mask_heights = np.random.randint(int(height * minimum_proportion), int(height * maximum_proportion), batch_size)
    mask_widths = np.random.randint(int(width * minimum_proportion), int(width * maximum_proportion), batch_size)
    
    # Random positions for the mask (ensuring mask fits within image boundaries)
    top_positions = np.random.randint(0, height - mask_heights + 1, batch_size)
    left_positions = np.random.randint(0, width - mask_widths + 1, batch_size)
    
    # Apply the masks and record coordinates
    for i in range(batch_size):
        top = top_positions[i]
        left = left_positions[i]
        mask_height = mask_heights[i]
        mask_width = mask_widths[i]
        
        # Define the coordinates
        bottom = top + mask_height
        right = left + mask_width
        
        # Set the mask area to -1 across all channels for the i-th image
        masked_images[i, :, top:bottom, left:right] = 255
        
        # Store the coordinates as (left, bottom, right, top)
        mask_coordinates[i] = [left, bottom, right, top]
    
    return masked_images, mask_coordinates

def mask_array_with_nan(arr, fraction=0.5):
    """
    Randomly mask a fraction of values in an array with NaN.

    Parameters:
    - arr: np.ndarray, the array to be masked
    - fraction: float, the fraction of values to be masked with NaN

    Returns:
    - masked_arr: np.ndarray, a copy of the array with NaN values injected
    """
    masked_arr = arr.copy()  # Create a copy to avoid modifying the original array
    total_values = arr.size  # Total number of elements in the array
    n_mask = int(total_values * fraction)  # Number of values to mask

    # Randomly choose indices to set as NaN
    flat_indices = np.random.choice(total_values, n_mask, replace=False)
    
    # Set chosen indices to NaN in the flattened array
    masked_arr.flat[flat_indices] = np.nan
    
    return masked_arr

def generate_random_id():
    #3 random 4 digit numbers separated by underscores
    n1 = np.random.randint(1e3, 1e4-1)
    n2 = np.random.randint(1e3, 1e4-1)
    n3 = np.random.randint(1e3, 1e4-1)
    return f"{n1}_{n2}_{n3}"

def validate_submission(final_submission):
    assert final_submission.dtype == np.uint8
    assert final_submission.shape == (1000, 3, 80, 128)
    minimum_val = np.min(final_submission)
    maximum_val = np.max(final_submission)
    if maximum_val<200 or minimum_val<0 or maximum_val>255:
        print(f"Warning: Your submission appears to be scaled between {minimum_val} and {maximum_val}. We expect the data to be scaled between 0-255. Are you sure everything is right?")
    else:
        print("Submission looks good!")