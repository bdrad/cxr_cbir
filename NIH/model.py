# from vit_pytorch.efficient import ViT
# from linformer import Linformer
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
from vit_pytorch import SimpleViT

def get_encoder(encoder_choice):
    if encoder_choice == 'vit':
        encoder = torch.hub.load('facebookresearch/deit:main',
                           'deit_tiny_patch16_224', pretrained=True)
        num_features = encoder.head.in_features
        encoder.head = nn.Linear(num_features, 2)
        return encoder