import numpy as np
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms

from models import *
from data_loader import *


def test_working(path, model, dataset_name):
    """model working test"""
    model.load_state_dict(torch.load(path)['model'])
    model.eval()
    print(f"acc: {torch.load(path)['acc']:.2%}")

    num_check_images = 10

    _, test_loader = data_loader(dataset_name, num_check_images)
    classes = data_classes(dataset_name)

    images, labels = next(iter(test_loader))
    output = model(images)
    pred = torch.max(output, 1).indices

    if dataset_name == 'cifar10':
        images[:, 0] = images[:, 0] * 0.2470 + 0.4914
        images[:, 1] = images[:, 1] * 0.2435 + 0.4822
        images[:, 2] = images[:, 2] * 0.2616 + 0.4465
        images = torchvision.utils.make_grid(images, nrow=num_check_images)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.imshow(np.transpose(images.numpy(), (1, 2, 0)))
        ax.text(-3, -10, 'actual: ', ha='right')
        ax.text(-3, -4, 'predicted: ', ha='right')
        for i in range(num_check_images):
            ax.text(18 + 34 * i, -10, classes[labels[i]], ha='center')
            ax.text(18 + 34 * i, -4, classes[pred[i]], ha='center')
        ax.axis('off')
        plt.show()

    else:
        # to be added
        pass


if __name__ == '__main__':
    # model load
    path = 'saved_models/best_VGG19_20220302_1232.pth'
    model = VGG(num_layers=19)
    test_working(path, model, 'cifar10')

