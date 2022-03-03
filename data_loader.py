import torch
import torchvision
import torchvision.transforms as transforms

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def data_loader(name, batch_size):
    """return both train_loader and test_loader"""

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*data_mu_sigma(name))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*data_mu_sigma(name))
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=f'dataset/{name}', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(
        root=f'dataset/{name}', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    return train_loader, test_loader


def data_mu_sigma(name):
    """return mean, std of dataset"""

    if name == 'cifar10':
        return [(0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)]

    else:
        # to be added
        return

def data_classes(name):
    """return classes of dataset"""

    if name == 'cifar10':
        return ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    else:
        # to be added
        return


if __name__ == '__main__':
    train_loader, test_loader = data_loader('cifar10', 10)

