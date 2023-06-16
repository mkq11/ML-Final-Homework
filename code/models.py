import utils
import numpy as np
import variable
import functional as F
import nn


class MLP(nn.Module):
    def __init__(
        self, input_size=784, num_classes=10, hidden_size=[64, 64], activation="relu"
    ):
        super(MLP, self).__init__()

        if activation == "relu":
            self.activation = F.relu
        elif activation == "sigmoid":
            self.activation = F.sigmoid
        else:
            raise NotImplementedError

        layer_size = [input_size] + hidden_size + [num_classes]
        self.layers = []
        for i in range(len(layer_size) - 1):
            layer = nn.Linear(layer_size[i], layer_size[i + 1])
            self.layers.append(layer)
            self.add_module(f"layer{i}", layer)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x


class MLPWithBatchNorm(nn.Module):
    def __init__(
        self, input_size=784, num_classes=10, hidden_size=[64, 64], activation="relu"
    ):
        super().__init__()

        if activation == "relu":
            self.activation = F.relu
        elif activation == "sigmoid":
            self.activation = F.sigmoid
        else:
            raise NotImplementedError

        layer_size = [input_size] + hidden_size + [num_classes]
        self.layers = []
        for i in range(len(layer_size) - 1):
            linear = nn.Linear(layer_size[i], layer_size[i + 1])
            if i != len(layer_size) - 2:
                bn = nn.BatchNorm1d(layer_size[i + 1])
                layer = nn.Sequential(linear, bn)
            else:
                layer = linear
            self.layers.append(layer)
            self.add_module(f"layer{i}", layer)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=10, num_blocks=[4, 8, 2]):
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.block1 = self.make_block(16, num_blocks=num_blocks[0])
        self.block2 = self.make_block(16, stride=2, num_blocks=num_blocks[1])
        self.block3 = self.make_block(32, stride=2, num_blocks=num_blocks[2])

        self.linear = nn.Linear(64, num_classes)

    def make_block(self, in_channels, stride=1, num_blocks=3):
        layers = []
        if stride != 1:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, in_channels * 2, kernel_size=1, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(in_channels * 2),
                )
            )
            in_channels *= 2
        for _ in range(num_blocks):
            layers.append(BasicBlock(in_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = out.mean(axis=(2, 3))
        out = self.linear(out)
        return out
