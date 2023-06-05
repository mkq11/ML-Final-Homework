import utils
import numpy as np
import variable
import functional as F
import nn
import optim


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


train_loader = utils.create_data_loader(batch_size=16)
test_loader = utils.create_data_loader(train=False)

network = ResNet(num_blocks=[4, 6, 2])
optimizer = optim.SGD(network.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

for epoch in range(20):
    network.train()
    train_loss = 0
    for i, (x, y) in enumerate(train_loader):
        y = variable.Variable(np.eye(10)[y])
        optimizer.zero_grad()
        out = network(x)
        loss = F.cross_entropy_loss(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.value
        acc = np.sum(np.argmax(out.value, axis=1) == np.argmax(y.value, axis=1))
        size = len(train_loader.dataset) // train_loader.batch_size
        print(f"{i:>{len(str(size))}} / {size}, acc = {acc:<3}", end="\r")
    train_loss /= len(train_loader.dataset)

    network.eval()
    acc = 0
    for x, y in test_loader:
        x = variable.Variable(x)
        out = network(x)
        acc += np.sum(np.argmax(out.value, axis=1) == y)
    acc /= len(test_loader.dataset)

    print(f"Epoch {epoch + 1}: Train loss {train_loss:.4e}, Test acc {100*acc:.2f}")
