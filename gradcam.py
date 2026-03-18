"""
gradcam.py
==========
Grad-CAM adapted for scalar regression (Equations 2–3 in paper).

Original Grad-CAM (Selvaraju et al. 2020) uses class scores.
Here the scalar ISN prediction ŷ replaces the class score.

    alpha_k = (1 / h'w') * Σ_{i,j}  ∂ŷ / ∂A^k_{ij}        (Eq. 2)

    L_Grad-CAM = ReLU( Σ_k  alpha_k * A^k )                  (Eq. 3)

The 7×7 activation map is bilinearly upsampled to 224×224 and
optionally overlaid on the original HMI image as a colour heatmap.

Usage:
  from gradcam import GradCAM
  gcam = GradCAM(model)                     # hooks last conv layer
  saliency = gcam.generate(img_tensor)      # (H, W) numpy array in [0,1]
  overlay  = gcam.overlay(img_tensor, hmi)  # (H, W, 3) uint8 RGB
"""

import numpy as np
import torch
import matplotlib as mpl
from PIL import Image as PILImage


class GradCAM:
    """
    Grad-CAM for scalar regression.

    Parameters
    ----------
    model        : ResNet50ISN  (or any model with a convolutional backbone)
    target_layer : nn.Module    (default: model.backbone.layer4[-1])
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()

        if target_layer is None:
            # Default: last residual block of ResNet50
            target_layer = model.backbone.layer4[-1]
        self.target_layer = target_layer

        self._activations: torch.Tensor = None   # type: ignore
        self._gradients:   torch.Tensor = None   # type: ignore

        self._register_hooks()

    # ── Hook registration ─────────────────────────────────────────────────────

    def _register_hooks(self) -> None:
        def fwd_hook(module, inp, output):
            # Save feature maps A^k  ∈ ℝ^{B × C × h' × w'}
            self._activations = output.detach()

        def bwd_hook(module, grad_in, grad_out):
            # Save gradients ∂ŷ/∂A^k
            self._gradients = grad_out[0].detach()

        # Store handles so hooks can be removed later
        self._fwd_handle = self.target_layer.register_forward_hook(fwd_hook)
        self._bwd_handle = self.target_layer.register_full_backward_hook(bwd_hook)

    def remove_hooks(self) -> None:
        """Remove registered hooks from the model."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    # ── Core Grad-CAM computation ─────────────────────────────────────────────

    def generate(self, img_tensor: torch.Tensor) -> np.ndarray:
        """
        Compute the Grad-CAM saliency map for one image.

        Parameters
        ----------
        img_tensor : torch.Tensor  shape (1, 3, H, W), normalised,
                     on the same device as the model

        Returns
        -------
        saliency : np.ndarray  shape (H, W), values in [0, 1]
        """
        device = next(self.model.parameters()).device
        x = img_tensor.to(device)

        # ── Forward pass ──────────────────────────────────────────────────────
        self.model.zero_grad()
        # Temporarily enable grad even if we're in eval/no_grad context
        with torch.enable_grad():
            x = x.requires_grad_(True)   # need grad graph
            pred = self.model(x)          # scalar ŷ per sample in batch
            # Backprop from scalar ŷ → Eq. 2
            pred.sum().backward()

        # ── Importance weights (Eq. 2) ────────────────────────────────────────
        # gradients : (1, C, h', w')
        # alpha_k   : spatial average over h'×w'  →  (1, C, 1, 1)
        alpha = self._gradients.mean(dim=(2, 3), keepdim=True)

        # ── Weighted activation map (Eq. 3) ──────────────────────────────────
        # activations : (1, C, h', w')
        cam = (alpha * self._activations).sum(dim=1).squeeze(0)   # (h', w')
        cam = torch.relu(cam).cpu().numpy()                        # ReLU

        # ── Normalise to [0, 1] ───────────────────────────────────────────────
        if cam.max() > 1e-8:
            cam = cam / cam.max()

        # ── Bilinear upsample to input resolution ─────────────────────────────
        H, W = img_tensor.shape[2], img_tensor.shape[3]
        cam_img = PILImage.fromarray((cam * 255).astype(np.uint8))
        cam_up  = np.array(
            cam_img.resize((W, H), PILImage.BILINEAR)
        ).astype(np.float32) / 255.0

        return cam_up   # (H, W) ∈ [0, 1]

    # ── Overlay helper ────────────────────────────────────────────────────────

    def overlay(
        self,
        img_tensor:    torch.Tensor,
        original_img:  np.ndarray,
        alpha:         float = 0.45,
        colormap:      str   = "jet",
    ) -> np.ndarray:
        """
        Blend the Grad-CAM heatmap with the original HMI image.

        Parameters
        ----------
        img_tensor   : (1, 3, H, W) normalised tensor
        original_img : H×W uint8 grayscale (or H×W×3 RGB) numpy array
        alpha        : heatmap blending weight  (0 = only image, 1 = only map)
        colormap     : matplotlib colormap name (default 'jet')

        Returns
        -------
        blended : (H, W, 3) uint8 RGB image
        """
        cam      = self.generate(img_tensor)                    # (H, W)
        heatmap  = mpl.colormaps[colormap](cam)[..., :3]       # (H, W, 3)

        if original_img.ndim == 2:
            base = np.stack([original_img / 255.0] * 3, axis=-1)
        else:
            base = original_img / 255.0

        blended = (1.0 - alpha) * base + alpha * heatmap
        blended = np.clip(blended, 0, 1)
        return (blended * 255).astype(np.uint8)

    # ── Batch saliency for a full DataLoader ──────────────────────────────────

    def generate_batch(
        self,
        loader,
        max_samples: int = 50,
    ) -> list:
        """
        Generate saliency maps for up to `max_samples` test images.

        Returns
        -------
        list of (saliency, isn_gt) tuples
        """
        results = []
        count   = 0
        for imgs, labels in loader:
            for i in range(len(imgs)):
                if count >= max_samples:
                    return results
                sal = self.generate(imgs[i:i+1])
                results.append((sal, float(labels[i])))
                count += 1
        return results
