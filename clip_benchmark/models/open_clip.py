import open_clip
import torch
from torch import nn


def vis_forward_wrapper(obj):
    def forward(x):
        x = obj.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [obj.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + obj.positional_embedding.to(x.dtype)
        x = obj.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = obj.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = obj.ln_post(x[:, 0, :])

        if obj.proj is not None:
            x = x @ obj.proj

        x = obj.lproj(x)

        return x

    return forward


def text_forward_wrapper(obj):
    def forward(x, attn_mask = None):
        for r in obj.resblocks:
                x = r(x, attn_mask=attn_mask)
        x = obj.lproj(x)
        return x

    return forward


def load_open_clip(model_name: str = "ViT-B-32-quickgelu", pretrained: str = "laion400m_e32", cache_dir: str = None, device="cpu"):
    model, _, transform = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, cache_dir=cache_dir)
    lrpoj = nn.Linear(512, 128)
    # Load state
    
    model.visual.lproj = lproj
    model.visual.forward = vis_forward_wrapper(model.visual)

    model.transformer.lproj = lproj
    model.transformer.forward = text_forward_wrapper(model.transformer)
    
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, transform, tokenizer
