import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=4),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=4), 
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc_layers = nn.Sequential(
            # cifar10 data: 32 * 32 images
            # Conv2d -> 32 + 2*4 - 5 + 1 = 36
            # MaxPool2d -> 36 / 2 = 18
            # Conv2d -> 18 + 2*4 - 5 + 1 = 22
            # MaxPool2d -> 22 / 2 = 11
            nn.Linear(32*11*11, 1024),
            nn.ReLU(), 
            nn.Linear(1024, 128), 
            nn.ReLU(), 
            nn.Linear(128, 32), 
            nn.ReLU(), 
            nn.Linear(32, 10)
        )
        self.dropout = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc_layers(x)
        return x

    def __str__(self):
        return 'CustomCNN'
