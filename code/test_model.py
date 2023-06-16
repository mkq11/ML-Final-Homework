import time

import numpy as np

import utils
import variable
import functional as F
import nn
import optim
import models


class TestDataset:
    def __init__(self):
        data, label = utils.read_test_data()
        self.data = data.reshape(-1, 28 * 28)
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


def train(model, train_loader, optimizer, loss_fn=F.cross_entropy_loss):
    model.train()
    train_loss = 0
    total_acc = 0
    for i, (x, y) in enumerate(train_loader):
        y = variable.Variable(np.eye(10)[y])
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.value
        acc = np.sum(np.argmax(out.value, axis=1) == np.argmax(y.value, axis=1))
        total_acc = total_acc + acc
        size = len(train_loader.dataset) // train_loader.batch_size
        print(f"{i:>{len(str(size))}} / {size}, acc = {acc:<3}", end="\r")
    train_loss /= len(train_loader.dataset)
    total_acc /= len(train_loader.dataset)
    return train_loss, total_acc


def test(model, test_loader, loss_fn=F.cross_entropy_loss):
    model.eval()
    test_loss = 0
    total_acc = 0
    for x, y in test_loader:
        x = variable.Variable(x)
        y = variable.Variable(np.eye(10)[y])
        out = model(x)
        loss = loss_fn(out, y)
        test_loss += loss.value
        acc = np.sum(np.argmax(out.value, axis=1) == np.argmax(y.value, axis=1))
        total_acc = total_acc + acc
    test_loss /= len(test_loader.dataset)
    total_acc /= len(test_loader.dataset)
    return test_loss, total_acc


def test_model(
    train_loader,
    val_loader,
    test_loader,
    model,
    loss_fn=F.cross_entropy_loss,
    num_epochs=20,
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,
):
    time_start = time.time()
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    test_loss, test_acc = test(model, val_loader, loss_fn)
    print(f"Epoch {0}: Test Loss = {test_loss:.4f}, Test Acc = {test_acc:.4f}")
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, loss_fn)
        test_loss, test_acc = test(model, val_loader, loss_fn)
        print(
            f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, "
            + f"Test Loss = {test_loss:.4f}, Test Acc = {test_acc:.4f}"
        )
    time_end = time.time()
    test_loss, test_acc = test(model, test_loader, loss_fn)
    print(f"Test Acc* = {test_acc * 10}")
    print(f"Time: {time_end - time_start:.2f}s")


def basic_test():
    train_loader = utils.create_data_loader(batch_size=64, flatten=True)
    val_loader = utils.create_data_loader(train=False, flatten=True)
    test_loader = utils.DataLoader(TestDataset(), batch_size=64)

    print(f"\n{'线性模型':=^80}")
    model = nn.Linear(784, 10)
    test_model(
        train_loader,
        val_loader,
        test_loader,
        model,
        loss_fn=F.mse_loss,
        momentum=0,
        weight_decay=0,
    )

    print(f"\n{'线性模型分类':=^80}")
    model = nn.Linear(784, 10)
    test_model(train_loader, val_loader, test_loader, model, momentum=0, weight_decay=0)

    print(f"\n{'多层感知机sigmoid':=^80}")
    model = models.MLP(784, 10, [64, 64], activation="sigmoid")
    test_model(
        train_loader, val_loader, test_loader, model, lr=0.5, momentum=0, weight_decay=0
    )

    print(f"\n{'多层感知机relu':=^80}")
    model = models.MLP(784, 10, [64, 64])
    test_model(train_loader, val_loader, test_loader, model, momentum=0, weight_decay=0)

    print(f"\n{'多层感知机+weight_decay':=^80}")
    model = models.MLP(784, 10, [64, 64])
    test_model(train_loader, val_loader, test_loader, model, momentum=0)

    print(f"\n{'多层感知机+momentum':=^80}")
    model = models.MLP(784, 10, [64, 64])
    test_model(train_loader, val_loader, test_loader, model)

    print(f"\n{'多层感知机+batchnorm':=^80}")
    model = models.MLPWithBatchNorm(784, 10, [64, 64])
    test_model(train_loader, val_loader, test_loader, model)


if __name__ == "__main__":
    basic_test()
