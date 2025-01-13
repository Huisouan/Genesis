import torch
from vector_quantize_pytorch import FSQ

quantizer = FSQ(
    levels = [8, 5, 5, 5]
)

x = torch.randn(1, 1024, 4) # 4 since there are 4 levels
xhat, indices = quantizer(x)
print(xhat.shape, indices.shape)
# (1, 1024, 4), (1, 1024)

assert torch.all(xhat == quantizer.indices_to_codes(indices))