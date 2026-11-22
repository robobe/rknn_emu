import torch
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from uav_data import CLASSES


def create_model(pretrained: bool = False):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT if pretrained else None)
    model.classifier[-1] = torch.nn.Linear(model.last_channel, len(CLASSES))
    return model
