import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassJaccardIndex

# --- Configuration Constant ---
NUM_CLASSES = 7

# --- Model Definition ---

def get_model(device, encoder_name='resnet101', encoder_weights='imagenet', classes=NUM_CLASSES):
    """
    Initializes the DeepLabV3Plus model for multi-class segmentation.
    """
    # CRITICAL CHANGE: classes=NUM_CLASSES (e.g., 7)
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=classes,    # Multi-channel output (e.g., 7 channels)
        activation=None     # raw logits for CrossEntropyLoss
    ).to(device)
    return model

# --- Loss Function ---

# CRITICAL CHANGE: Initialize multi-class components
# nn.CrossEntropyLoss expects logits [B, C, H, W] and targets [B, H, W] of type Long
BCE = nn.CrossEntropyLoss(ignore_index=6) # Use ignore_index if 'Unlabelled' is common
# DiceLoss for multi-class. mode='multiclass' is essential.
DICE = smp.losses.DiceLoss(mode='multiclass', classes=NUM_CLASSES, ignore_index=6) 

def combined_loss(preds, targets):
    """
    Weighted combination of CrossEntropy and Dice Loss for multi-class training.
    """
    # Targets must be LongTensor [B, H, W] for both.
    return 0.6 * BCE(preds, targets) + 0.4 * DICE(preds, targets)

# --- Metric ---

# Use a PyTorch Metric for robust multi-class mIoU calculation
mIoU_metric = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average='macro', ignore_index=6)

def iou_score(preds, targets, num_classes=NUM_CLASSES):
    """
    Computes Mean IoU (Jaccard Index) for the multi-class prediction using torchmetrics.
    """
    # preds are logits [B, C, H, W]
    # targets are LongTensor [B, H, W]
    
    # Ensure targets are on the correct device for the metric calculation
    targets = targets.cpu() 
    preds = preds.cpu()
    
    # torchmetrics expects preds (logits) and targets (integer class labels)
    # The output is a single tensor value (the mIoU across all classes)
    miou = mIoU_metric(preds, targets).item()
    return miou

# --- Example Usage (Not run in script) ---
if __name__ == '__main__':
    print("Model utility file ready for Multi-Class (7 classes).")