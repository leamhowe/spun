import numpy as np
import torch


def intersection_over_union(pred, true):
    """
    Calculates intersection and union for a batch of images.

    Args:
        pred (torch.Tensor): a tensor of predictions
        true (torc.Tensor): a tensor of labels

    Returns:
        intersection (int): total intersection of pixels
        union (int): total union of pixels
    """
    valid_pixel_mask = true.ne(255)  # valid pixel mask
    true = true.masked_select(valid_pixel_mask).to("cpu")
    pred = pred.masked_select(valid_pixel_mask).to("cpu")

    # Intersection and union totals
    intersection = np.logical_and(true, pred)
    union = np.logical_or(true, pred)
    return intersection.sum() / union.sum()

def class_specific_iou(preds, labels, class_id=2):
    # Assumes preds and labels are already tensors, typically after argmax on the logits if necessary
    # Create masks for the specific class
    pred_mask = (preds == class_id)
    true_mask = (labels == class_id)

    # Calculate intersection and union
    intersection = (pred_mask & true_mask).sum()
    union = (pred_mask | true_mask).sum()

    if union == 0:
        return 0.0
    else:
        return intersection / union


def dice_score(preds, targets, epsilon=1e-6):
    # Assuming preds and targets are binary tensors of shape (batch_size, ...)
    preds = preds.contiguous()
    targets = targets.contiguous()        

    intersection = (preds & targets).float().sum((1, 2, 3))  # Assume NCHW format
    union = preds.float().sum((1, 2, 3)) + targets.float().sum((1, 2, 3))
    
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.mean()  # Return mean Dice score over batch


def f1_score(preds, targets, epsilon=1e-6):
    # Assuming preds and targets are binary tensors of shape (batch_size, ...)
    preds = preds.contiguous()
    targets = targets.contiguous()        

    intersection = (preds & targets).float().sum((1, 2, 3))  # Assume NCHW format
    precision = intersection / (preds.float().sum((1, 2, 3)) + epsilon)
    recall = intersection / (targets.float().sum((1, 2, 3)) + epsilon)
    
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1.mean()  # Return mean F1 score over batch

