import torch
import timm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = ["COVID", "Lung_Opacity", "Normal", "Viral_Pneumonia"]

def load_single_model(path: str):
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=4
    )
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False))
    model.to(DEVICE)
    model.eval()
    return model

def load_pgd_model(path: str):
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=4
    )
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model

def load_all_models():
    return {
        "standard": load_single_model("models/vit_standard.pth"),
        "pgd":      load_pgd_model("models/vit_pgd.pth"),
        "fgsm":     load_single_model("models/vit_fgsm.pth"),
    }