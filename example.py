import torch
from swin_transformer_pytorch import SwinTransformer
from upernet import UperNet


uperNet = UperNet()


dummy_x = torch.randn(2, 3, 256, 256)
out = uperNet(dummy_x)


print(out)