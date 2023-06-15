from typing import OrderedDict
import numpy as np

import variable
import functional as F


class Paramerter(variable.Variable):
    def __init__(self, shape, empty=False):
        if empty:
            super().__init__(np.empty(shape))
        else:
            super().__init__(np.random.randn(*shape) * 0.01)
        self.requires_grad = True


class Module:
    def __init__(self) -> None:
        super().__setattr__("_parameters", OrderedDict())
        super().__setattr__("_modules", OrderedDict())
        super().__setattr__("training", True)

    def train(self):
        self.training = True
        for module in self._modules.values():
            module.train()
        for param in self._parameters.values():
            param.requires_grad = True

    def eval(self):
        self.training = False
        for module in self._modules.values():
            module.eval()
        for param in self._parameters.values():
            param.requires_grad = False

    def forward(self, x):
        raise NotImplementedError

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = variable.Variable(x)
        return self.forward(x)

    def named_parameters(self):
        for name, param in self._parameters.items():
            yield name, param
        for name, module in self._modules.items():
            for n, param in module.named_parameters():
                yield f"{name}.{n}", param

    def parameters(self):
        for name, param in self.named_parameters():
            yield param

    def __setattr__(self, name: str, value) -> None:
        super().__setattr__(name, value)
        param = self.__dict__.get("_parameters")
        modules = self.__dict__.get("_modules")

        if param is None or modules is None:
            raise AttributeError(
                "cannot assign parameters before Module.__init__() call"
            )
        if name in param:
            del param[name]
        if name in modules:
            del modules[name]

        if isinstance(value, Module):
            modules[name] = value
        elif isinstance(value, Paramerter):
            param[name] = value


class Sequential(Module):
    def __init__(self, *args) -> None:
        super().__init__()
        for idx, module in enumerate(args):
            self.__setattr__(f"layer{idx}", module)
        self.layers_num = len(args)

    def forward(self, x):
        for idx in range(self.layers_num):
            x = self.__getattribute__(f"layer{idx}")(x)
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = Paramerter((in_features, out_features), empty=True)
        self.bias = Paramerter((out_features,), empty=True)
        self.reset_parameters()

    def reset_parameters(self):
        e = 1 / np.sqrt(self.weight.value.shape[0])
        self.weight.value = np.random.uniform(-e, e, self.weight.value.shape)
        self.bias.value = np.zeros(self.bias.value.shape)

    def forward(self, x):
        return x @ self.weight + self.bias


class BatchNorm1d(Module):
    def __init__(self, num_features: int, momentum=0.1, eps=1e-5) -> None:
        super().__init__()
        self.num_features = num_features
        self.weight = Paramerter((num_features,))
        self.bias = Paramerter((num_features,))
        self.running_mean = np.zeros((num_features,), dtype=np.float32)
        self.running_var = np.ones((num_features,), dtype=np.float32)
        self.momentum = momentum
        self.eps = eps
        self.weight.value = np.ones((num_features,), dtype=np.float32)
        self.bias.value = np.zeros((num_features,), dtype=np.float32)

    def forward(self, x):
        if self.training:
            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * batch_mean.value
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * batch_var.value
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var
        x = (x - batch_mean) / ((batch_var + self.eps) ** 0.5)
        return x * self.weight + self.bias


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.weight = Paramerter((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = Paramerter((out_channels,))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)


class BatchNorm2d(Module):
    def __init__(self, num_features: int, momentum=0.1, eps=1e-5) -> None:
        super().__init__()
        self.num_features = num_features
        self.weight = Paramerter((num_features,))
        self.bias = Paramerter((num_features,))
        self.running_mean = np.zeros((1, num_features, 1, 1), dtype=np.float32)
        self.running_var = np.ones((1, num_features, 1, 1), dtype=np.float32)
        self.momentum = momentum
        self.eps = eps
        self.weight.value = np.ones((num_features,), dtype=np.float32)
        self.bias.value = np.zeros((num_features,), dtype=np.float32)

    def forward(self, x: variable.Variable):
        if self.training:
            batch_mean = x.mean(axis=(0, 2, 3), keepdims=True)
            batch_var = x.var(axis=(0, 2, 3), keepdims=True)
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * batch_mean.value
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * batch_var.value
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var
        x = (x - batch_mean) / ((batch_var + self.eps) ** 0.5)
        x = x * self.weight.reshape(1, -1, 1, 1) + self.bias.reshape(1, -1, 1, 1)
        return x
