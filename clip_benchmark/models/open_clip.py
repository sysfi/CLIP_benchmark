import open_clip
import torch

def load_open_clip(model_name: str = "ViT-B-32-quickgelu", pretrained: str = "laion400m_e32", cache_dir: str = None, device="cpu"):
    model, _, transform = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = torch.quantization.quantize_dynamic(
        model, dtype=torch.quint8
    )
    model = model.to(device)
    tokenizer = open_clip.tokenizer.tokenize
    return model, transform, tokenizer
