import torch
import torch.nn as nn
import torch.nn.functional as F


architecture_dict = {
    18: ('Block', [2, 2, 2, 2]),
    34: ('Block', [3, 4, 6, 3]),
    50: ('BottleneckBlock', [3, 4, 6, 3]),
    101: ('BottleneckBlock', [3, 4, 23, 3]),
    152: ('BottleneckBlock', [3, 8, 36, 3])
}
initial_channels_each_layer = [64, 128, 256, 512]


class Block(nn.Module):
    def __init__(self, block_in, block_out, block_stride):
        super(Block, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(block_in, block_out, kernel_size=3, stride=block_stride, padding=1, bias=False),
            nn.BatchNorm2d(block_out),
            nn.ReLU(),
            nn.Conv2d(block_out, block_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(block_out)
        )

        if block_stride != 1:  # projection shortcut
            self.shortcut = nn.Sequential(
                nn.Conv2d(block_in, block_out, kernel_size=1, stride=block_stride, bias=False), # ignore half of the pixels
                nn.BatchNorm2d(block_out)
            )
        else:  # identity shortcut
            self.shortcut = nn.Sequential()

        self.block_in = block_in
        self.block_out = block_out

    def forward(self, x):
        out = self.layers(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BottleneckBlock(nn.Module):
    def __init__(self, block_in, block_inside, block_stride, expand=4):
        super(BottleneckBlock, self).__init__()

        block_out = block_inside * expand

        self.layers = nn.Sequential(
            nn.Conv2d(block_in, block_inside, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(block_inside),
            nn.ReLU(),
            nn.Conv2d(block_inside, block_inside, kernel_size=3, stride=block_stride, padding=1, bias=False),
            nn.BatchNorm2d(block_inside),
            nn.ReLU(),
            nn.Conv2d(block_inside, block_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(block_out),
        )

        if block_stride != 1 or block_in != block_out:  # projection shortcut
            self.shortcut = nn.Sequential(
                nn.Conv2d(block_in, block_out, kernel_size=1, stride=block_stride, bias=False),  # ignore half of the pixels
                nn.BatchNorm2d(block_out)
            )
        else:  # identity shortcut
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.layers(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_layers=34):
        super(ResNet, self).__init__()
        self.num_layers = num_layers

        block_type, num_blocks = architecture_dict[num_layers]

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.last_block_out = 64
        self.conv2 = self.build_layers(block_type, num_blocks[0], initial_channels_each_layer[0], 1)
        self.conv3 = self.build_layers(block_type, num_blocks[1], initial_channels_each_layer[1], 2)
        self.conv4 = self.build_layers(block_type, num_blocks[2], initial_channels_each_layer[2], 2)
        self.conv5 = self.build_layers(block_type, num_blocks[3], initial_channels_each_layer[3], 2)

        if block_type == 'Block':
            channels_after_conv = initial_channels_each_layer[-1]
        else:  # block_type == 'BottleneckBlock'
            channels_after_conv = initial_channels_each_layer[-1] * 4

        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.fc_layer = nn.Linear(channels_after_conv, 10)

    def build_layers(self, block_type, num_blocks, num_channels, first_stride):
        strides = [first_stride] + [1]*(num_blocks-1)
        layers = []
        if block_type == 'Block':
            for stride in strides:
                layers.append(Block(block_in=self.last_block_out, block_out=num_channels, block_stride=stride))
                self.last_block_out = num_channels
        else:  # block_type == 'BottleneckBlock'
            for stride in strides:
                layers.append(BottleneckBlock(block_in=self.last_block_out, block_inside=num_channels,
                                              block_stride=stride, expand=4))
                self.last_block_out = num_channels * 4
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_layer(x)
        return x

    def __str__(self):
        return f'{self.__class__.__name__}{self.num_layers}'


if __name__ == '__main__':
    model = ResNet(num_layers=152)
    sample = torch.randn(1, 3, 32, 32)
    # print(model)
    print(model(sample))
