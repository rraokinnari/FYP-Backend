import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
from Utils.loader import LABELS, DEVICE

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def bytes_to_tensor(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return TRANSFORM(img).unsqueeze(0).to(DEVICE)

def run_inference(tensor, model):
    with torch.no_grad():
        logits = model(tensor)
    probs = F.softmax(logits, dim=1)[0]
    pred_idx = probs.argmax().item()
    return {
        "predicted_class": LABELS[pred_idx],
        "confidence_scores": {
            label: round(probs[i].item(), 4)
            for i, label in enumerate(LABELS)
        }
    }