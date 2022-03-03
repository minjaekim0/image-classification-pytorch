import torch
import torch.nn as nn
import torch.optim as optim

from models import *
from data_loader import data_loader
from trainer import Trainer


if __name__ == '__main__':
    model_list = [
        # (model, batch_size, num_epochs)

        # (CustomCNN(), 512, 5),
        (VGG(num_layers=19), 256, 200),
        (ResNet(num_layers=152), 64, 200),
    ]

    for model, batch_size, num_epochs in model_list:
        train_loader, test_loader = data_loader('cifar10', batch_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        tr = Trainer(train_loader, test_loader, num_epochs, model, criterion, optimizer, scheduler)
        tr.run()
        tr.plot()
