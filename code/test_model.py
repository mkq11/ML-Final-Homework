import utils
import numpy as np
import variable
import functional as F
import nn
import optim


class Network(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


train_loader = utils.create_data_loader()
test_loader = utils.create_data_loader(False)

network = Network()
optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

for epoch in range(10):
    train_loss = 0
    for x, y in train_loader:
        x = variable.Variable(x)
        y = variable.Variable(np.eye(10)[y])
        optimizer.zero_grad()
        out = network.forward(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.value
    train_loss /= len(train_loader.dataset)

    acc = 0
    for x, y in test_loader:
        x = variable.Variable(x)
        out = network.forward(x)
        acc += np.sum(np.argmax(out.value, axis=1) == y)
    acc /= len(test_loader.dataset)

    print(f"Epoch {epoch + 1}: Train loss {train_loss:.4e}, Test acc {100*acc:.2f}")
