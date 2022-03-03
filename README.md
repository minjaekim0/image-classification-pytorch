# image-classification-pytorch

### implemented models
- VGG
- ResNet

### preprocessing
- normalization of each RGB layer of images
- data augmentation: crop, flip

### result for CIFAR10
- \# of epochs = 200
- Adam optimizer: lr = 1e-3, weight_decay = 1e-4

| model      | accuracy |
|------------|----------|
| VGG-19     | 89.85%   |
| ResNet-152 | 90.48%   |