import torch

from backend.models.segformer_model import build_segformer


model = build_segformer(num_classes=5)

dummy = torch.randn(1, 3, 512, 512)

with torch.no_grad():
    outputs = model(pixel_values=dummy)

print("Logits shape:", outputs.logits.shape)