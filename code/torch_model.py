import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import utils


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
    

class MNISTDataset(Dataset):
    def __init__(self, train=True):
        df = pd.read_csv("./data/mnist_data.csv")
        if train:
            df = df.iloc[:5000]
        else:
            df = df.iloc[5000:7000]
        self.data = df.iloc[:, 1:].values / 255
        self.data = self.data.reshape(-1, 1, 28, 28)
        self.label = df.iloc[:, 0].values
        self.data = torch.from_numpy(self.data).float()
        self.label = torch.from_numpy(self.label).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]
    

class TestDataset(Dataset):
    def __init__(self):
        data, label = utils.read_test_data()
        self.data = torch.from_numpy(data).float()
        self.label = torch.from_numpy(label).long()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    acc = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        acc += torch.sum(torch.argmax(out, axis=1) == y).item()
    train_loss /= len(train_loader.dataset)
    acc /= len(train_loader.dataset)
    return train_loss, acc

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    acc = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        test_loss += loss.item()
        acc += torch.sum(torch.argmax(out, axis=1) == y).item()
    test_loss /= len(test_loader.dataset)
    acc /= len(test_loader.dataset)
    return test_loss, acc

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("使用CPU训练，时间较长")

    train_loader = DataLoader(MNISTDataset(train=True), batch_size=64, shuffle=True)
    val_loader = DataLoader(MNISTDataset(train=False), batch_size=64, shuffle=False)
    test_loader = DataLoader(TestDataset(), batch_size=64, shuffle=False)

    model = ResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(100):
        _, train_acc = train(model, train_loader, optimizer, criterion, device)
        _, val_acc = test(model, val_loader, criterion, device)
        print("Epoch: {}, Train Acc: {}, Val Acc: {}".format(epoch, train_acc, val_acc))
        scheduler.step()

    _, test_acc = test(model, test_loader, criterion, device)
    print("Test Acc: {}".format(test_acc))
