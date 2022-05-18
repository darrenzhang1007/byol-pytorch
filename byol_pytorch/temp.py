import torch
import torchvision.models as models

import torchvision.models as models
import torch
from models.mlp_head import MLPHead


class ResNet18(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet18, self).__init__()
        # if kwargs['name'] == 'resnet18':
            # resnet = models.resnet18(pretrained=False)
        # elif kwargs['name'] == 'resnet50':
        resnet = models.resnet50(pretrained=False)

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        # self.projetion = MLPHead(in_channels=resnet.fc.in_features, **kwargs['projection_head'])

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        # return self.projetion(h)


load_params = torch.load(r'F:\WorkSpace\class\pretrained_model\resnet50-0676ba61.pth')
# online_network.load_state_dict(load_params['online_network_state_dict'])
# print(load_params)
online_network = ResNet18('resnet50')

online_network.load_state_dict(load_params)