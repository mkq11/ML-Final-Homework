import utils
import numpy as np
import variable
import functional as F


class Network:
    def __init__(self) -> None:
        self.w1 = variable.Variable(np.random.randn(784, 128) * 0.01)
        self.b1 = variable.Variable(np.zeros(128))
        self.w2 = variable.Variable(np.random.randn(128, 64) * 0.01)
        self.b2 = variable.Variable(np.zeros(64))
        self.w3 = variable.Variable(np.random.randn(64, 10) * 0.01)
        self.b3 = variable.Variable(np.zeros(10))

    def forward(self, x):
        x = F.add(F.mul(x, self.w1), self.b1)
        x = F.relu(x)
        x = F.add(F.mul(x, self.w2), self.b2)
        x = F.relu(x)
        x = F.add(F.mul(x, self.w3), self.b3)
        return x

    def parameters(self):
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]


train_loader = utils.create_data_loader()
test_loader = utils.create_data_loader(False)

network = Network()

for epoch in range(10):
    train_loss = 0
    for x, y in train_loader:
        x = variable.Variable(x)
        y = variable.Variable(np.eye(10)[y])
        for param in network.parameters():
            param.zero_grad()
        out = network.forward(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        for param in network.parameters():
            param.value -= param.grad * 0.01
        train_loss += loss.value
    train_loss /= len(train_loader.dataset)

    acc = 0
    for x, y in test_loader:
        x = variable.Variable(x)
        out = network.forward(x)
        acc += np.sum(np.argmax(out.value, axis=1) == y)
    acc /= len(test_loader.dataset)

    print(f"Epoch {epoch + 1}: Train loss {train_loss:.4f}, Test acc {100*acc:.2f}")
