import torch
import torch.nn.functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def _get_clamp_bounds(device):
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
    std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)
    lower = (0 - mean) / std
    upper = (1 - mean) / std
    return lower, upper

def fgsm_attack(tensor, model, epsilon=0.01):
    images = tensor.clone().detach()
    images.requires_grad_(True)

    outputs = model(images)
    loss = F.cross_entropy(outputs, outputs.argmax(dim=1))

    model.zero_grad()
    loss.backward()

    data_grad = images.grad.detach()
    adv_images = images + epsilon * data_grad.sign()

    lower, upper = _get_clamp_bounds(tensor.device)
    adv_images = torch.clamp(adv_images, lower, upper)

    return adv_images.detach()

def apply_attack(tensor, attack_type: str, model, epsilon=0.01):
    if attack_type == "fgsm":
        return fgsm_attack(tensor, model, epsilon)
    return tensor