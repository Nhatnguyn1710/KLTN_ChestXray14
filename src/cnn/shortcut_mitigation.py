
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional, Dict


# ============================================================
# 1. Anatomy Mask Generator - Tạo mask vùng phổi/tim từ ảnh
# ============================================================
class AnatomyMaskGenerator:
    """
    Tạo approximate anatomy masks cho chest X-ray.
    
    Trong production nên dùng segmentation model (VD: U-Net trained on JSRT/SIIM).
    Đây là phiên bản heuristic đơn giản dựa trên histogram và morphology.
    """
    
    @staticmethod
    def generate_lung_mask(image_np: np.ndarray, lung_ratio: float = 0.6) -> np.ndarray:
        """
        Tạo approximate lung mask từ chest X-ray.
        
        Args:
            image_np: RGB image normalized [0, 1], shape [H, W, 3]
            lung_ratio: Tỉ lệ height/width expected cho vùng phổi
            
        Returns:
            Binary mask [H, W] where 1 = lung region
        """
        h, w = image_np.shape[:2]
        
        # Convert to grayscale
        if image_np.ndim == 3:
            gray = (image_np.mean(axis=2) * 255).astype(np.uint8)
        else:
            gray = (image_np * 255).astype(np.uint8)
        
        # Phổi thường là vùng tối hơn (air absorbs less X-ray)
        # Adaptive threshold để tìm vùng tối
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations để làm sạch
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Approximate lung region: middle-upper phần ảnh
        lung_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Left lung: khoảng 10-45% width, 15-75% height
        left_lung = np.zeros((h, w), dtype=np.uint8)
        left_lung[int(h*0.15):int(h*0.75), int(w*0.10):int(w*0.45)] = 255
        
        # Right lung: khoảng 55-90% width, 15-75% height
        right_lung = np.zeros((h, w), dtype=np.uint8)
        right_lung[int(h*0.15):int(h*0.75), int(w*0.55):int(w*0.90)] = 255
        
        # Combine với threshold mask
        lung_mask = cv2.bitwise_and(cleaned, left_lung | right_lung)
        
        return (lung_mask > 0).astype(np.float32)
    
    @staticmethod
    def generate_heart_mask(image_np: np.ndarray) -> np.ndarray:
        """
        Tạo approximate cardiac silhouette mask.
        Tim thường ở center-left, 30-65% height từ top.
        """
        h, w = image_np.shape[:2]
        
        # Approximate heart region
        heart_mask = np.zeros((h, w), dtype=np.float32)
        
        # Heart: khoảng 35-65% width (center-left), 30-65% height
        heart_mask[int(h*0.30):int(h*0.65), int(w*0.35):int(w*0.65)] = 1.0
        
        # Làm mềm edges
        heart_mask = cv2.GaussianBlur(heart_mask, (31, 31), 0)
        
        return heart_mask
    
    @staticmethod
    def generate_anatomy_mask(
        image_np: np.ndarray, 
        include_lungs: bool = True,
        include_heart: bool = True,
    ) -> np.ndarray:
        """
        Tạo combined anatomy mask cho chest X-ray.
        
        Returns:
            Soft mask [H, W] in [0, 1] where high = anatomy regions
        """
        h, w = image_np.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        
        if include_lungs:
            lung_mask = AnatomyMaskGenerator.generate_lung_mask(image_np)
            mask = np.maximum(mask, lung_mask)
        
        if include_heart:
            heart_mask = AnatomyMaskGenerator.generate_heart_mask(image_np)
            mask = np.maximum(mask, heart_mask)
        
        return mask


# ============================================================
# 2. Anatomy-Guided Attention Loss
# ============================================================
class AnatomyGuidedLoss(nn.Module):
    """
    Penalize model khi CAM/attention focus vào vùng ngoài anatomy.
    
    L_anatomy = -mean(CAM * anatomy_mask) + λ * mean(CAM * (1 - anatomy_mask))
    
    Vế 1: Khuyến khích CAM sáng ở vùng anatomy
    Vế 2: Phạt CAM sáng ở vùng ngoài anatomy
    """
    
    def __init__(
        self, 
        outside_penalty: float = 2.0,
        inside_bonus: float = 1.0,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.outside_penalty = outside_penalty
        self.inside_bonus = inside_bonus
        self.smooth = smooth
    
    def forward(
        self, 
        cam: torch.Tensor,           # [B, H, W] attention/CAM maps
        anatomy_mask: torch.Tensor,  # [B, H, W] binary anatomy mask
    ) -> torch.Tensor:
        """
        Args:
            cam: Grayscale CAM hoặc attention map, normalized [0, 1]
            anatomy_mask: Binary mask where 1 = valid anatomy region
            
        Returns:
            Scalar loss
        """
        # Normalize CAM to [0, 1] if needed
        cam_min = cam.amin(dim=(1, 2), keepdim=True)
        cam_max = cam.amax(dim=(1, 2), keepdim=True)
        cam_norm = (cam - cam_min) / (cam_max - cam_min + self.smooth)
        
        # Inside anatomy: want high activation
        inside = cam_norm * anatomy_mask
        inside_mean = inside.sum() / (anatomy_mask.sum() + self.smooth)
        
        # Outside anatomy: penalize high activation
        outside_mask = 1.0 - anatomy_mask
        outside = cam_norm * outside_mask
        outside_mean = outside.sum() / (outside_mask.sum() + self.smooth)
        
        loss = -self.inside_bonus * inside_mean + self.outside_penalty * outside_mean
        
        return loss


# ============================================================
# 3. Background Invariance Regularization
# ============================================================
class BackgroundInvarianceLoss(nn.Module):
    """
    Regularization để model invariant với background changes.
    
    Ý tưởng: Với cùng 1 vùng anatomy, thay đổi background không nên 
    thay đổi prediction.
    
    L_bg = MSE(logits_original, logits_bg_replaced)
    """
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
    
    def forward(
        self,
        logits_original: torch.Tensor,  # [B, C] original predictions
        logits_augmented: torch.Tensor, # [B, C] predictions với background thay đổi
    ) -> torch.Tensor:
        """
        Forces predictions to be similar regardless of background.
        """
        return self.weight * F.mse_loss(logits_original, logits_augmented)


# ============================================================
# 4. CAM-Guided Regularization (Right for Right Reasons)
# ============================================================
class RightForRightReasonsLoss(nn.Module):
    """
    Regularization theo paper "Right for the Right Reasons".
    
    Penalize model khi gradient w.r.t. input cao ở vùng không liên quan.
    L_rrr = ||grad_input * (1 - anatomy_mask)||_2
    """
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
    
    def forward(
        self,
        input_grad: torch.Tensor,    # [B, 3, H, W] gradient của loss w.r.t. input
        anatomy_mask: torch.Tensor,  # [B, 1, H, W] anatomy mask
    ) -> torch.Tensor:
        """
        Penalize gradients outside anatomy region.
        """
        outside_mask = 1.0 - anatomy_mask
        outside_grad = input_grad * outside_mask
        
        # L2 norm của gradient ngoài anatomy
        loss = torch.sqrt((outside_grad ** 2).sum(dim=(1, 2, 3)) + 1e-8).mean()
        
        return self.weight * loss


# ============================================================
# 5. Multi-Domain Mixup (chống domain shift)
# ============================================================
class MultiDomainMixup:
    """
    Mixup images từ các domains/hospitals khác nhau.
    
    Giúp model không overfit vào đặc điểm của 1 domain cụ thể
    (VD: hospital-specific markers, scanner characteristics).
    """
    
    @staticmethod
    def mixup_domains(
        x1: torch.Tensor,           # [B, C, H, W] images từ domain 1
        y1: torch.Tensor,           # [B, num_classes] labels từ domain 1
        x2: torch.Tensor,           # [B, C, H, W] images từ domain 2
        y2: torch.Tensor,           # [B, num_classes] labels từ domain 2
        alpha: float = 0.4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mix images và labels từ 2 domains.
        """
        lam = np.random.beta(alpha, alpha)
        
        # Simple interpolation
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_y = lam * y1 + (1 - lam) * y2
        
        return mixed_x, mixed_y


# ============================================================
# 6. Attention Entropy Regularization
# ============================================================
class AttentionEntropyLoss(nn.Module):
    """
    Regularize attention/CAM để không quá sparse hoặc quá diffuse.
    
    - Quá sparse → có thể focus vào shortcut pixel
    - Quá diffuse → không học được feature cụ thể
    
    Target: moderate entropy
    """
    
    def __init__(
        self, 
        target_entropy: float = 0.5,  # Target entropy (normalized)
        weight: float = 0.1,
    ):
        super().__init__()
        self.target_entropy = target_entropy
        self.weight = weight
    
    def forward(self, attention: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attention: [B, H, W] normalized attention maps in [0, 1]
        """
        # Flatten spatial dimensions
        B = attention.shape[0]
        attn_flat = attention.view(B, -1)
        
        # Normalize to probability distribution
        attn_sum = attn_flat.sum(dim=1, keepdim=True) + 1e-8
        attn_prob = attn_flat / attn_sum
        
        # Compute entropy
        log_prob = torch.log(attn_prob + 1e-8)
        entropy = -(attn_prob * log_prob).sum(dim=1)
        
        # Normalize by max possible entropy
        max_entropy = np.log(attn_flat.shape[1])
        norm_entropy = entropy / max_entropy
        
        # Penalize deviation from target
        loss = ((norm_entropy - self.target_entropy) ** 2).mean()
        
        return self.weight * loss


# ============================================================
# 7. Gradient Reversal Layer (for domain adaptation)
# ============================================================
class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal cho Domain Adversarial Training.
    
    Forward: identity
    Backward: negate gradients
    
    Dùng với domain discriminator để học domain-invariant features.
    """
    
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class DomainAdversarialHead(nn.Module):
    """
    Domain discriminator head cho domain-adversarial training.
    
    Giúp model học features không phụ thuộc vào domain (hospital, scanner, etc.)
    """
    
    def __init__(self, in_features: int, num_domains: int = 2, hidden_dim: int = 256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_domains),
        )
    
    def forward(self, features: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        Args:
            features: [B, D] feature vectors
            alpha: Gradient reversal strength
            
        Returns:
            Domain logits [B, num_domains]
        """
        reversed_features = GradientReversalLayer.apply(features, alpha)
        return self.classifier(reversed_features)


# ============================================================
# 8. Cutout với Anatomy Awareness
# ============================================================
class AnatomyAwareCutout:
    """
    Cutout augmentation nhưng KHÔNG cutout vùng anatomy quan trọng.
    
    Thay vì random cutout có thể che mất tổn thương,
    chỉ cutout vùng background/borders.
    """
    
    @staticmethod
    def apply(
        image: torch.Tensor,          # [B, C, H, W]
        anatomy_mask: torch.Tensor,   # [B, 1, H, W] 
        num_holes: int = 4,
        hole_size_ratio: float = 0.1,
        p: float = 0.5,
    ) -> torch.Tensor:
        """
        Apply cutout chỉ ở vùng NGOÀI anatomy.
        """
        if np.random.random() > p:
            return image
        
        B, C, H, W = image.shape
        hole_h = int(H * hole_size_ratio)
        hole_w = int(W * hole_size_ratio)
        
        image_out = image.clone()
        
        for b in range(B):
            for _ in range(num_holes):
                # Random position
                y = np.random.randint(0, H - hole_h)
                x = np.random.randint(0, W - hole_w)
                
                # Check if mostly outside anatomy
                mask_region = anatomy_mask[b, 0, y:y+hole_h, x:x+hole_w]
                if mask_region.mean() < 0.3:  # Mostly outside anatomy
                    image_out[b, :, y:y+hole_h, x:x+hole_w] = 0
        
        return image_out


# ============================================================
# 9. Combined Anti-Shortcut Loss
# ============================================================
class AntiShortcutLoss(nn.Module):
    """
    Combined loss để chống shortcut learning.
    
    L_total = L_classification 
              + λ_anatomy * L_anatomy_guided
              + λ_bg * L_background_invariance
              + λ_entropy * L_attention_entropy
    """
    
    def __init__(
        self,
        anatomy_weight: float = 0.1,
        bg_invariance_weight: float = 0.05,
        entropy_weight: float = 0.02,
        target_entropy: float = 0.5,
    ):
        super().__init__()
        
        self.anatomy_loss = AnatomyGuidedLoss(
            outside_penalty=2.0,
            inside_bonus=1.0,
        )
        self.bg_loss = BackgroundInvarianceLoss(weight=bg_invariance_weight)
        self.entropy_loss = AttentionEntropyLoss(
            target_entropy=target_entropy,
            weight=entropy_weight,
        )
        
        self.anatomy_weight = anatomy_weight
    
    def forward(
        self,
        classification_loss: torch.Tensor,
        cam: Optional[torch.Tensor] = None,
        anatomy_mask: Optional[torch.Tensor] = None,
        logits_original: Optional[torch.Tensor] = None,
        logits_augmented: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined anti-shortcut loss.
        
        Returns:
            Dict with individual losses and total
        """
        losses = {"classification": classification_loss}
        total = classification_loss
        
        # Anatomy-guided loss
        if cam is not None and anatomy_mask is not None:
            l_anatomy = self.anatomy_weight * self.anatomy_loss(cam, anatomy_mask)
            losses["anatomy_guided"] = l_anatomy
            total = total + l_anatomy
            
            # Entropy regularization on CAM
            l_entropy = self.entropy_loss(cam)
            losses["entropy"] = l_entropy
            total = total + l_entropy
        
        # Background invariance
        if logits_original is not None and logits_augmented is not None:
            l_bg = self.bg_loss(logits_original, logits_augmented)
            losses["bg_invariance"] = l_bg
            total = total + l_bg
        
        losses["total"] = total
        return losses


# ============================================================
# Usage Example
# ============================================================
"""
Tích hợp vào training loop:

from src.cnn.shortcut_mitigation import (
    AnatomyMaskGenerator,
    AntiShortcutLoss,
    AnatomyAwareCutout,
)

# Initialize
anatomy_gen = AnatomyMaskGenerator()
anti_shortcut = AntiShortcutLoss(
    anatomy_weight=0.1,
    bg_invariance_weight=0.05,
    entropy_weight=0.02,
)

# In training loop:
for images, labels in dataloader:
    # Generate anatomy masks (có thể cache hoặc dùng pre-computed)
    anatomy_masks = []
    for img in images:
        mask = anatomy_gen.generate_anatomy_mask(img.numpy())
        anatomy_masks.append(mask)
    anatomy_masks = torch.tensor(anatomy_masks)
    
    # Forward
    logits = model(images)
    
    # Get CAM (optional, có thể chỉ dùng 1 số batches)
    if use_cam_supervision:
        cam = compute_gradcam(model, images, logits)
    else:
        cam = None
    
    # Classification loss
    cls_loss = criterion(logits, labels)
    
    # Anti-shortcut loss
    losses = anti_shortcut(
        classification_loss=cls_loss,
        cam=cam,
        anatomy_mask=anatomy_masks,
    )
    
    total_loss = losses["total"]
    total_loss.backward()
    optimizer.step()
"""
