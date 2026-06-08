
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from torchvision import models

try:
    import torchxrayvision as xrv
    HAS_XRV = True
except ImportError:
    HAS_XRV = False


class PCAMPool(nn.Module):
    """
    Position-Calibrated Activation Map pooling (CheXpert Top1 - jfhealthcare).
    Học attention map từ feature map, weighted average pooling thay vì GAP.
    Giúp model tập trung vào vùng quan trọng thay vì gộp đều toàn ảnh.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.attention = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.attention.weight)
        nn.init.constant_(self.attention.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        attn = torch.sigmoid(self.attention(x))          # [B, 1, H, W]
        weighted = x * attn                              # [B, C, H, W]
        pool = weighted.flatten(2).sum(-1) / (attn.flatten(2).sum(-1) + 1e-10)  # [B, C]
        return pool


class LSEPool(nn.Module):
    """Log-Sum-Exp pooling (Pinheiro & Collobert, 2015).

    LSE_r(F) = (1/r) * log( (1/HW) * sum_ij exp(r * F_ij) )

    r > 1: focuses on spatial peaks (closer to max pooling).
    r → 0: approaches average pooling (GAP).
    r is a learnable scalar — adapts during training.

    Better than GAP for multi-label X-ray: highlights focal pathology
    (Nodule, Mass, Pneumothorax) without the attention-collapse issues of PCAM.
    Grad-CAM gradients through LSE are spatially informative — no need to
    switch to GAP like we do for PCAM.
    """
    def __init__(self, r_init: float = 5.0):
        super().__init__()
        self.r = nn.Parameter(torch.tensor(r_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        r = self.r.clamp(min=0.1, max=20.0)   # keep LSE numerically tame
        x_flat = x.flatten(2)       # [B, C, H*W]
        # Numerically stable: subtract per-(B,C) max before exp
        x_max = x_flat.max(dim=2, keepdim=True).values   # [B, C, 1]
        exp_x = torch.exp(r * (x_flat - x_max))          # [B, C, H*W]
        lse = x_max.squeeze(2) + torch.log(exp_x.mean(dim=2)) / r  # [B, C]
        return lse


class DenseNet121MultiLabel(nn.Module):
    """
    DenseNet-121 với classifier head cho multi-label classification.

    - Backbone: DenseNet-121 pretrained trên ImageNet
    - Head: Linear(1024[+1], 512) → BatchNorm/LayerNorm → ReLU → Dropout
            → Linear(512, num_classes)  (logits, sigmoid áp khi inference)
    - v5: Hỗ trợ inject view_position (AP/PA) vào features trước head
    - v6: PCAM pooling (Position-Calibrated Activation Maps) thay GAP
    - Output: logits cho mỗi nhãn bệnh (independent multi-label)
    """

    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        dropout: float = 0.3,
        use_view_position: bool = False,
        use_pcam: bool = False,
        use_lse: bool = False,
        xrv_pretrained: str = None,
        head_norm: str = "layernorm",
        use_lung_mask: bool = False,
        thorax_mask_cfg: dict = None,
    ):
        super(DenseNet121MultiLabel, self).__init__()
        self.use_view_position = use_view_position
        self.head_norm = str(head_norm or "layernorm").strip().lower()
        # v12: configurable ellipse params (default = original hardcoded values)
        _tm = thorax_mask_cfg if isinstance(thorax_mask_cfg, dict) else {}
        self._mask_cx = float(_tm.get("cx", 0.50))
        self._mask_cy = float(_tm.get("cy", 0.42))
        self._mask_sx = float(_tm.get("sx", 0.30))
        self._mask_sy = float(_tm.get("sy", 0.28))

        # Load DenseNet-121 backbone
        if xrv_pretrained and HAS_XRV:
            # torchxrayvision pretrained: trained on 800k+ X-ray images
            # (NIH + CheXpert + MIMIC-CXR + PadChest + RSNA + SIIM + VinBrain)
            # Much better features for chest X-ray than ImageNet.
            print(f"  [model] Loading torchxrayvision backbone: {xrv_pretrained}")
            xrv_model = xrv.models.DenseNet(weights=xrv_pretrained)
            # XRV uses 1-channel input; our pipeline uses 3-channel RGB.
            # Replicate the 1-channel conv weight across 3 channels (÷3 to preserve scale).
            self.backbone = models.densenet121(weights=None)
            xrv_state = xrv_model.features.state_dict()
            first_conv_key = "conv0.weight"  # [64, 1, 7, 7] in XRV
            if first_conv_key in xrv_state:
                w1ch = xrv_state[first_conv_key]  # [64, 1, 7, 7]
                xrv_state[first_conv_key] = w1ch.repeat(1, 3, 1, 1) / 3.0  # [64, 3, 7, 7]
            self.backbone.features.load_state_dict(xrv_state, strict=False)
            print("  [model] XRV backbone loaded (1ch->3ch adapted)")
        elif pretrained:
            weights = models.DenseNet121_Weights.IMAGENET1K_V1
            self.backbone = models.densenet121(weights=weights)
        else:
            self.backbone = models.densenet121(weights=None)

        # Lấy số features từ classifier gốc
        in_features = self.backbone.classifier.in_features  # 1024
        hidden_dim = 512

        # Xóa backbone classifier — sẽ gọi features thủ công trong forward()
        self.backbone.classifier = nn.Identity()

        # v6: PCAM pooling
        self.use_pcam = use_pcam
        if use_pcam:
            self.pcam_pool = PCAMPool(in_features)

        # LSE pooling (non-parametric spatial aggregation, safer than PCAM for NIH)
        self.use_lse = use_lse and not use_pcam  # PCAM takes priority if both set
        if self.use_lse:
            self.lse_pool = LSEPool(r_init=5.0)

        # v-lungmask: Body mask on feature map to prevent learning shortcuts
        # (devices, text markers, background artifacts).
        self.use_lung_mask = use_lung_mask

        # Optional AP/PA hint override used only during Grad-CAM pass.
        self._gradcam_view_type_override = None

        # v5: Nếu dùng view_position thì head input = 1024 + 1 = 1025
        head_in = in_features + 1 if use_view_position else in_features
        head_layers = [nn.Linear(head_in, hidden_dim)]
        if self.head_norm == "batchnorm":
            head_layers.append(nn.BatchNorm1d(hidden_dim))
        elif self.head_norm == "layernorm":
            head_layers.append(nn.LayerNorm(hidden_dim))
        elif self.head_norm not in {"none", "identity"}:
            raise ValueError(f"Unsupported head_norm: {head_norm}")
        head_layers.extend(
            [
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, num_classes),
                # v9: Removed nn.Sigmoid() — model outputs raw logits.
                # Sigmoid is applied inside AsymmetricLossLogits for numerical stability
                # (log-sum-exp trick avoids log(0) and log(1) issues).
                # For inference, apply torch.sigmoid() on outputs explicitly.
            ]
        )
        self.classifier_head = nn.Sequential(*head_layers)

        # Khởi tạo weights cho head mới (Kaiming init)
        self._init_classifier()

    def _init_classifier(self):
        """Khởi tạo weights cho classifier head (Kaiming)."""
        for m in self.classifier_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _compute_body_mask(self, x: torch.Tensor, feat_h: int, feat_w: int) -> torch.Tensor:
        """Compute anatomical lung ROI mask at feature map resolution.

        OLD approach (brightness threshold): kept shoulders, clavicle, wires,
        text markers, and all peripheral confounders — only removed true black
        background. Model could still see AP indicators, support devices, etc.

        NEW approach combines:
        1. Brightness foreground detection (remove true black background/borders)
        2. Anatomical prior: soft elliptical ROI centered on thoracic cavity
           (~20-80% width, ~12-70% height)
        3. Peripheral suppression: fade out shoulder/clavicle region (top 12%)

        At 7x7 feature resolution (224px input), this effectively zeros out:
        - Corners (text markers "L", "R", "SUPINE PORT", "AP")
        - Top strip (shoulder/clavicle region — common false attention area)
        - Side margins (arms, body wall, edge artifacts)
        While keeping the central thorax (lungs + heart + mediastinum).
        """
        with torch.no_grad():
            B, C, H, W = x.shape
            gray = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]

            # Step 1: Foreground detection (remove true black borders)
            thresh = gray.flatten(2).mean(dim=2, keepdim=True).unsqueeze(-1)
            fg_mask = (gray > thresh * 0.5).float()  # [B, 1, H, W]

            # Step 2: Anatomical prior — soft elliptical ROI on thorax
            # Lungs + heart occupy roughly center 60% width, top 12-70% height
            y_grid = torch.linspace(0, 1, H, device=x.device).view(1, 1, H, 1)
            x_grid = torch.linspace(0, 1, W, device=x.device).view(1, 1, 1, W)

            # Thorax center & semi-axes (v12: read from config via instance attrs)
            cx, cy = self._mask_cx, self._mask_cy
            sx, sy = self._mask_sx, self._mask_sy

            dist_sq = ((x_grid - cx) / sx) ** 2 + ((y_grid - cy) / sy) ** 2
            # Smooth falloff: 1.0 inside ellipse, 0.0 well outside
            anatomy_prior = torch.clamp(1.5 - dist_sq, 0.0, 1.0)  # [1, 1, H, W]

            # Step 3: Suppress top strip (shoulder/clavicle = top ~12%)
            top_rows = max(1, int(H * 0.12))
            top_fade = torch.linspace(0.15, 1.0, top_rows, device=x.device)
            anatomy_prior[:, :, :top_rows, :] *= top_fade.view(1, 1, top_rows, 1)

            # Combine: foreground AND anatomical prior
            combined = fg_mask * anatomy_prior

            # Downsample to feature map resolution
            combined = F.adaptive_avg_pool2d(combined, (feat_h, feat_w))  # [B, 1, h, w]
            # Normalize so max = 1.0 per sample
            max_val = combined.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
            combined = combined / max_val

        return combined  # [B, 1, h, w]

    def forward(self, x: torch.Tensor, view_type: torch.Tensor = None) -> torch.Tensor:
       
        # Trích xuất features thủ công — cần thiết để inject view_position
        feats = self.backbone.features(x)        # [B, 1024, h, w]
        feats = F.relu(feats, inplace=False)      # inplace=False: preserve original tensor for Grad-CAM hooks

        # v-lungmask: Zero out feature activations outside body region.
        # Forces PCAM attention + classifier to learn from thorax only.
        # Apply both train AND eval to avoid train-test distribution mismatch.
        if self.use_lung_mask:
            body_mask = self._compute_body_mask(x, feats.shape[2], feats.shape[3])
            feats = feats * body_mask  # [B, 1024, h, w] × [B, 1, h, w]

        if self.use_pcam:
            feats = self.pcam_pool(feats)         # [B, 1024] via PCAM
        elif self.use_lse:
            feats = self.lse_pool(feats)          # [B, 1024] via LSE pooling
        else:
            feats = F.adaptive_avg_pool2d(feats, (1, 1))
            feats = torch.flatten(feats, 1)       # [B, 1024]

        # v5: Concat view position vào features trước head
        if self.use_view_position:
            if view_type is None and self._gradcam_view_type_override is not None:
                view_type = self._gradcam_view_type_override
            if view_type is not None:
                vt = view_type
                if vt.ndim == 0:
                    vt = vt.view(1, 1)
                elif vt.ndim == 1:
                    vt = vt.unsqueeze(1)
                elif vt.ndim > 2:
                    vt = vt.view(vt.shape[0], -1)
                if vt.shape[1] != 1:
                    vt = vt[:, :1]
                if vt.shape[0] != feats.shape[0]:
                    if vt.shape[0] == 1:
                        vt = vt.expand(feats.shape[0], 1)
                    else:
                        vt = vt[:feats.shape[0], :]
                vt = vt.to(feats.device).float()
            else:
                # Không có view info → dùng 0.5 (unknown)
                vt = torch.full((feats.shape[0], 1), 0.5, device=feats.device)
            feats = torch.cat([feats, vt], dim=1)  # [B, 1025]

        return self.classifier_head(feats)

    @contextmanager
    def gradcam_mode(self):
        """Temporarily switch to GAP for GradCAM visualization.

        Both PCAM and LSE create non-uniform gradient flow back to the
        feature map, which makes Grad-CAM scattered/dotty:
          - PCAM: attention weights bias gradients toward learned attention peaks.
          - LSE (r large):  ∂LSE/∂F_ij ≈ softmax(r·F_ij)  →  gradient mass
            collapses onto a few spatial peaks → "chấm li ti" heatmap.

        Switching to GAP gives uniform spatial gradient (1/HW per position),
        which produces smooth, interpretable blob heatmaps equivalent to
        Stanford CheXNet's CAM.  The convolutional features are identical —
        only the pooling pathway used during the visualization pass changes.
        Inference (and metrics) outside this context still uses LSE/PCAM.
        """
        original_pcam = self.use_pcam
        original_lse = self.use_lse
        self.use_pcam = False
        self.use_lse = False
        try:
            yield
        finally:
            self.use_pcam = original_pcam
            self.use_lse = original_lse

    @contextmanager
    def gradcam_view_context(self, view_type: torch.Tensor = None):
        """Temporarily inject view_type for Grad-CAM forward(path = model(x))."""
        original = self._gradcam_view_type_override
        if view_type is None:
            self._gradcam_view_type_override = None
            try:
                yield
            finally:
                self._gradcam_view_type_override = original
            return

        vt = view_type.detach()
        if vt.ndim == 0:
            vt = vt.view(1)
        model_device = next(self.parameters()).device
        self._gradcam_view_type_override = vt.to(model_device)
        try:
            yield
        finally:
            self._gradcam_view_type_override = original

    @property
    def features(self):
        """Truy cập feature extractor (dùng cho Grad-CAM)."""
        return self.backbone.features

    # ---- Freeze / Unfreeze backbone ----

    def freeze_backbone(self):
       
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        
        for param in self.backbone.features.parameters():
            param.requires_grad = True

    def get_param_groups(self, lr_backbone: float, lr_head: float) -> list:
        
        head_params = list(self.classifier_head.parameters())
        if self.use_pcam:
            head_params += list(self.pcam_pool.parameters())
        if self.use_lse:
            head_params += list(self.lse_pool.parameters())
        return [
            {"params": self.backbone.features.parameters(), "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ]

    def count_params(self) -> dict:
        """Đếm số params total / trainable / frozen."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {"total": total, "trainable": trainable, "frozen": frozen}


def build_model(config: dict) -> DenseNet121MultiLabel:
  
    cnn_cfg = config["cnn"]
    model = DenseNet121MultiLabel(
        num_classes=cnn_cfg["num_classes"],
        pretrained=cnn_cfg["pretrained"],
        dropout=cnn_cfg.get("dropout", 0.3),
        use_view_position=cnn_cfg.get("use_view_position", False),  # v5
        use_pcam=cnn_cfg.get("use_pcam", False),                    # v6
        use_lse=cnn_cfg.get("use_lse", False),                      # LSE pooling
        xrv_pretrained=cnn_cfg.get("xrv_pretrained", None),        # v14: torchxrayvision backbone
        head_norm=cnn_cfg.get("head_norm", "layernorm"),
        use_lung_mask=cnn_cfg.get("use_lung_mask", False),
        thorax_mask_cfg=cnn_cfg.get("thorax_mask", None),          # v12: configurable ellipse
    )
    return model


def load_trained_model(checkpoint_path: str, config: dict, device: torch.device) -> DenseNet121MultiLabel:
    """
    Load model đã train từ checkpoint.
    
    Args:
        checkpoint_path: Đường dẫn file .pth
        config: dict config
        device: torch.device
    Returns:
        Model đã load weights, ở eval mode
    """
    model = build_model(config)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # v5 compatibility: checkpoint cũ dùng "backbone.classifier.*"
    # model mới dùng "classifier_head.*" — remap tự động
    if any(k.startswith("backbone.classifier.") for k in state_dict):
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("backbone.classifier."):
                new_k = k.replace("backbone.classifier.", "classifier_head.", 1)
                remapped[new_k] = v
            else:
                remapped[k] = v
        state_dict = remapped
        print("  [model] Remapped backbone.classifier.* -> classifier_head.* (v5 compat)")

    # Nếu checkpoint có head 1024-dim nhưng model mới dùng 1025-dim (view_position),
    # rebuild model không có view_position để load weights cũ
    head_weight_key = "classifier_head.0.weight"
    if head_weight_key in state_dict:
        ckpt_in_features = state_dict[head_weight_key].shape[1]  # 1024 hoặc 1025
        if ckpt_in_features == 1024 and model.use_view_position:
            print("  [model] Checkpoint trained without view_position (1024-dim head).")
            print("  [model] Loading with use_view_position=False. Retrain to use view_position.")
            from copy import deepcopy
            cfg_copy = deepcopy(config)
            cfg_copy["cnn"]["use_view_position"] = False
            model = build_model(cfg_copy)

    if any(k.startswith("pcam_pool.") for k in state_dict) and not model.use_pcam:
        print("  [model] Checkpoint uses PCAM pooling; loading in compatibility mode.")
        from copy import deepcopy
        cfg_copy = deepcopy(config)
        cfg_copy["cnn"]["use_pcam"] = True
        cfg_copy["cnn"]["use_view_position"] = model.use_view_position
        cfg_copy["cnn"]["head_norm"] = model.head_norm
        model = build_model(cfg_copy)

    if "lse_pool.r" in state_dict and not model.use_lse:
        print("  [model] Checkpoint uses LSE pooling; loading in compatibility mode.")
        from copy import deepcopy
        cfg_copy = deepcopy(config)
        cfg_copy["cnn"]["use_lse"] = True
        cfg_copy["cnn"]["use_pcam"] = False
        cfg_copy["cnn"]["use_view_position"] = model.use_view_position
        cfg_copy["cnn"]["head_norm"] = model.head_norm
        model = build_model(cfg_copy)

    # Old checkpoints used BatchNorm1d in the head. If current config switched to
    # LayerNorm/Identity, rebuild the compatibility model so historical runs can
    # still be inspected without forcing the new architecture onto old weights.
    if "classifier_head.1.running_mean" in state_dict and model.head_norm != "batchnorm":
        print("  [model] Checkpoint uses BatchNorm head; loading in compatibility mode.")
        from copy import deepcopy
        cfg_copy = deepcopy(config)
        cfg_copy["cnn"]["head_norm"] = "batchnorm"
        cfg_copy["cnn"]["use_view_position"] = model.use_view_position
        model = build_model(cfg_copy)

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model
