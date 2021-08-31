import torch.nn as nn
import torch
from torchvision import models as models


# def model(pretrained, requires_grad, num_classes):
#     model = models.resnet18(progress=True, pretrained=pretrained)
#     # to freeze the hidden layers
#     if not requires_grad:
#         for param in model.parameters():
#             param.requires_grad = False
#     # to train the hidden layers
#     else:
#         for param in model.parameters():
#             param.requires_grad = True
#     # make the classification layer learnable
#     # we have 22 classes in total
#     model.fc = nn.Linear(512, num_classes * 2)
#
#     return model


def load_model_from_checkpoint(checkpoint_pth, model):
    pretrained_dict = torch.load(checkpoint_pth, map_location=torch.device('cpu'))
    pretrained_dict_1 = {k[9:]: v for k, v in pretrained_dict['model'].items() if k[9:] in model.state_dict()}

    model.load_state_dict(pretrained_dict_1)
    return model


class resnet(nn.Module):
    def __init__(self, num_classes, checkpoint_pth=None, freeze=False, pretrained=False):
        super().__init__()
        self.backbone = models.resnet18(zero_init_residual=True, pretrained=pretrained) # image net,
        self.backbone.fc = nn.Identity()
        if checkpoint_pth:
            self.backbone = load_model_from_checkpoint(checkpoint_pth, self.backbone)
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.backbone.fc = nn.Linear(512, num_classes * 2)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.shape[0], self.num_classes, 2)
        return x
