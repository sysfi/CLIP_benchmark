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
    def encode_text(text):
        x = obj.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + obj.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = obj.transformer(x, attn_mask=obj.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = obj.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ obj.text_projection
        
        x = obj.lproj(x)
        
        return x
    return encode_text

def load_open_clip(model_name: str = "ViT-B-32-quickgelu", pretrained: str = "laion400m_e32", cache_dir: str = None, device="cpu"):
    model, _, transform = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    
    lproj = nn.Linear(512, 128)
    # Load state
    
    lproj = nn.Linear(512, 128)
    model.visual.lproj = lproj
    model.visual.forward = vis_forward_wrapper(model.visual)
    model.lproj = lproj
    model.encode_text = text_forward_wrapper(model)
    
    model = model.to(device)
    tokenizer = open_clip.tokenizer.tokenize
    return model, transform, tokenizer
