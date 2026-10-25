import torch
import numpy as np
import matplotlib.cm as cm
from PIL import Image
import base64, io

def mask_patch_at(image, x, y, patch_size=32):
    masked = image.clone()
    _, H, W = masked.shape
    x1 = max(0, x - patch_size // 2)
    x2 = min(W, x + patch_size // 2)
    y1 = max(0, y - patch_size // 2)
    y2 = min(H, y + patch_size // 2)
    masked[:, y1:y2, x1:x2] = 0
    return masked

def generate_mda_heatmap(tensor, model, patch_size=32, stride=16):
    """
    tensor: shape [1, 3, 224, 224] — already preprocessed
    model: timm ViT model in eval mode
    Returns: base64-encoded PNG with heatmap overlaid on the original image
    """
    device = tensor.device
    image = tensor.squeeze(0)  # [3, 224, 224]

    # baseline confidence — use predicted class since we have no true label
    with torch.no_grad():
        base_out = model(tensor)
        pred_class = base_out.argmax(dim=1).item()
        base_prob = torch.softmax(base_out, dim=1)[0, pred_class].item()

    _, H, W = image.shape
    heatmap = np.zeros((H, W))

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            masked_img = mask_patch_at(image, x, y, patch_size)
            masked_tensor = masked_img.unsqueeze(0)

            with torch.no_grad():
                out = model(masked_tensor)
                prob = torch.softmax(out, dim=1)[0, pred_class].item()

            drop = base_prob - prob
            heatmap[y:y+stride, x:x+stride] = drop

    # normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # convert original image tensor to PIL Image
    image_np = image.cpu().numpy()  # [3, 224, 224]
    image_np = np.transpose(image_np, (1, 2, 0))  # [224, 224, 3]
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
    original_img = Image.fromarray(image_np)

    # apply jet colormap to heatmap
    colored_heatmap = cm.jet(heatmap)
    heatmap_img = Image.fromarray(np.uint8(colored_heatmap * 255))
    heatmap_img = heatmap_img.convert("RGB")  # ensure RGB mode

    # overlay heatmap on original image with 50% alpha blending
    overlay = Image.blend(original_img, heatmap_img, alpha=0.5)

    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")