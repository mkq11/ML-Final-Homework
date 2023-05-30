from ast import mod
from typing import OrderedDict
import numpy as np

import variable
import functional as F


class Paramerter(variable.Variable):
    def __init__(self, shape):
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


class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = Paramerter((in_features, out_features))
        self.bias = Paramerter((out_features,))

    def forward(self, x):
        return x @ self.weight + self.bias
