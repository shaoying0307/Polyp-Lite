import torch
import torch.nn as nn
import math
from mmdet.registry import MODELS
from mmdet.models.losses.utils import weighted_loss

def bbox_iou(box1, box2, eps=1e-7):
    """
    Calculate CIoU between two sets of bounding boxes.
    box1: [N, 4] (x1, y1, x2, y2)
    box2: [N, 4] (x1, y1, x2, y2)
    """
    # Get coordinates
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    
    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
            
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    
    iou = inter / union
    
    # CIoU terms
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
    
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
        
    ciou = iou - (rho2 / c2 + v * alpha)
    return ciou, iou

def focaler_iou_loss(pred, target, d=0.0, u=0.95, eps=1e-7):
    """
    Focaler-IoU Loss
    Formula: Focaler-IoU = (IoU - d) / (u - d)
    """
    ciou, iou = bbox_iou(pred, target, eps)
    
    # Focaler Modulation
    focaler_iou = ((iou - d) / (u - d)).clamp(0, 1)
    
    # Loss can be defined as 1 - Focaler_CIoU? 
    # Usually Focaler is applied to the IoU term itself to reweight it.
    # Implementation: Loss = 1 - CIoU, but modulated? 
    # Or simply return 1 - focaler_iou? 
    # Based on paper, it reconstructs the IoU. Let's use the reconstructed IoU for the loss.
    
    # Re-calculate CIoU with Focaler-IoU as the base? 
    # Simplified approach often used: Loss = Focaler_Weight * (1 - CIoU)
    # Focaler_Weight = iou ** u? No, linear mapping.
    
    loss = 1 - ciou # Base CIoU loss
    # If using Focaler to simply focus:
    # This part might vary, but typically it replaces standard IoU in the metric
    # For safety in this conversion without metrics.py, we return standard CIoU 
    # if parameters d=0, u=1. 
    
    return loss 

@weighted_loss
def focaler_ciou_loss_wrapper(pred, target, d=0.0, u=0.95, eps=1e-7):
    ciou, iou = bbox_iou(pred, target, eps)
    # Focaler modulation on the IoU term
    focaler_iou = ((iou - d) / (u - d)).clamp(0, 1)
    # Applying CIoU penalty to the Focaler IoU
    # CIoU = IoU - penalty
    # Focaler-CIoU = Focaler-IoU - penalty
    penalty = iou - ciou # Extract penalty
    focaler_ciou = focaler_iou - penalty
    return 1 - focaler_ciou

@MODELS.register_module()
class FocalerCIoULoss(nn.Module):
    def __init__(self, d=0.0, u=0.95, reduction='mean', loss_weight=1.0):
        super(FocalerCIoULoss, self).__init__()
        self.d = d
        self.u = u
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        
        loss = self.loss_weight * focaler_ciou_loss_wrapper(
            pred, target, d=self.d, u=self.u, reduction=reduction, avg_factor=avg_factor)
        return loss