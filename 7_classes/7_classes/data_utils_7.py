import numpy as np
import cv2
from torch.utils.data import Dataset
import os
from glob import glob
import torch

# --- Configuration Placeholder for Multi-Class Mapping ---

# Define the 7-class map from RGB to Integer ID (0-6)
# NOTE: The index 6 is explicitly used for 'Unlabelled' pixels, which can be ignored in loss.
CLASS_MAPPING_RGB_TO_ID = {
    # Class 0: Informal Settlements (was 1 in user list)
    (250, 235, 185): 0, 
    # Class 1: Built-Up (was 2)
    (200, 200, 200): 1, 
    # Class 2: Impervious Surfaces (was 3)
    (100, 100, 150): 2, 
    # Class 3: Vegetation (was 4) -> Monitor for Deforestation/Growth
    (80, 140, 50):   3, 
    # Class 4: Barren (was 5)
    (200, 160, 40):  4, 
    # Class 5: Water (was 6) -> Monitor for Waterbody Dynamics
    (40, 120, 240):  5, 
    # Class 6: Unlabelled (was 7) -> Often 0,0,0, set to 6 for ignore_index
    (0, 0, 0):       6, 
}

# --- DIP Pre-Processing Functions (enhance_image_dip remains the same) ---

def enhance_image_dip(img_bgr):
    """
    Applies CLAHE and Gaussian Blur for image enhancement.
    """
    # 1. Noise Reduction
    blurred_bgr = cv2.BilateralBlur(img_bgr, (3, 3), 0)
    # 2. Contrast Enhancement (CLAHE)
    lab = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    final_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return final_bgr

# --- Mask Conversion Function (NEW: RGB to Multi-Class ID) ---

def rgb_mask_to_multiclass_id(mask_rgb):
    """
    Converts a multi-class RGB mask to a single-channel mask with integer class IDs (0-6).
    Inputs: mask_rgb: HxWx3 numpy uint8 (RGB)
    Outputs: HxW numpy uint8 (0 to 6)
    """
    H, W, _ = mask_rgb.shape
    # Initialize mask with the ID for the first color (or a default ID)
    mask_id = np.full((H, W), fill_value=CLASS_MAPPING_RGB_TO_ID[(0,0,0)], dtype=np.uint8) 

    # Iterate through all known RGB values and map them to their integer ID
    for rgb_value, class_id in CLASS_MAPPING_RGB_TO_ID.items():
        if rgb_value == (0, 0, 0): continue # Skip the default background
        
        # Create a boolean mask where all three channels match the current rgb_value
        match_mask = np.all(mask_rgb == np.array(rgb_value, dtype=np.uint8), axis=-1)
        
        # Apply the class_id only where there is a match
        mask_id[match_mask] = class_id
        
    return mask_id # HxW, type uint8 (0-6)

# --- PyTorch Dataset ---

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        # Assuming filenames match between images and masks
        image_paths = sorted(glob(os.path.join(image_dir, "*.tif"))) 
        mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))   # Assuming masks are PNGs
        
        assert len(image_paths) == len(mask_paths)
        
        self.images = image_paths
        self.masks = mask_paths
        self.transform = transform
        
    def __len__(self): 
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Load BGR image and mask 
        img_bgr = cv2.imread(self.images[idx])
        mask_bgr = cv2.imread(self.masks[idx])
        
        # 2. DIP Image Enhancement 
        img_enhanced_bgr = enhance_image_dip(img_bgr)
        img_enhanced_rgb = img_enhanced_bgr[:,:,::-1] # BGR->RGB for model input
        
        # 3. Multi-Class Mask Conversion (CRITICAL CHANGE)
        mask_rgb = mask_bgr[:,:,::-1] # Convert mask to RGB first
        mask_id = rgb_mask_to_multiclass_id(mask_rgb)  # HxW -> 0-6 integer IDs
        
        mask_id = mask_id[..., None] # H,W,1 for albumentations compatibility
        
        # 4. Augmentation/Normalization 
        if self.transform:
            augmented = self.transform(image=img_enhanced_rgb, mask=mask_id)
            img, mask = augmented['image'], augmented['mask'] # mask is [1, H, W]
            
        # CRITICAL CHANGE: Mask must be LongTensor (integer IDs) for CrossEntropyLoss
        # Squeeze the channel dimension if ToTensorV2 leaves it as [1, H, W]
        return img, mask.squeeze(0).long() # Returns img [3, H, W] and mask [H, W] of type Long