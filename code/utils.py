import pandas as pd
import numpy as np
import matplotlib.image as mpimg


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index = 0
        self.data_size = len(dataset)
        self.indices = np.arange(self.data_size)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.data_size:
            raise StopIteration
        batch_indices = self.indices[self.index : self.index + self.batch_size]
        batch = [self.dataset[i] for i in batch_indices]
        self.index += self.batch_size
        data = np.array([x[0] for x in batch])
        label = np.array([x[1] for x in batch])
        return data, label


class MNISTDataset:
    def __init__(self, type="train"):
        df = pd.read_csv("./data/mnist_data.csv")
        if type == "train":
            df = df.iloc[:5000]
        elif type == "val":
            df = df.iloc[5000:7000]
        else:
            df = df.iloc[5000:]
        self.data = df.iloc[:, 1:].values / 255
        self.data = self.data.reshape(-1, 1, 28, 28)
        self.label = df.iloc[:, 0].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


def create_data_loader(batch_size=64, train=True):
    dataset = MNISTDataset('train' if train else 'val')
    loader = DataLoader(dataset, batch_size, train)
    return loader


def read_test_data():
    data = []
    label = []
    for i in range(10):
        img = mpimg.imread(f"./data/test_data/{i}.jpg")
        img = img[:, :, 1].reshape(28, 28)
        img = 1 - img / 255
        data.append(img)
        label.append(i)
    data = np.array(data).reshape(-1, 1, 28, 28)
    label = np.array(label)
    return data, label