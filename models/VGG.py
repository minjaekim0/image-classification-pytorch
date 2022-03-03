import torch
import torch.nn as nn


config_dict = {
    11: [64, 'M', 128, 'M', 256, 256, 'M',
         512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M',
         512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
         512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
         512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):
    def __init__(self, num_layers=16):
        super(VGG, self).__init__()
        self.num_layers = num_layers

        config = config_dict[num_layers]
        layers = []
        in_channels = 3

        for x in config:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, x, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(x))
                layers.append(nn.ReLU())
                in_channels = x
        self.layers = nn.Sequential(*layers)

        self.fc_layers = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

        # In the paper, there is three fully-connected layers(4096, 4096, 1000 channels each)
        # and two dropout layers between themselves.
        # However I use CIFAR10, so images' size just before fully-connected layer is 512*1*1 = 512
        # so dimensions are changed properly.

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x

    def __str__(self):
        return f'{self.__class__.__name__}{self.num_layers}'