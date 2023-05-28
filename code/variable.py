from typing import Any
import numpy as np


class Variable:
    def __init__(self, x):
        self.value = np.array(x, dtype=np.float32)
        self.inputs = []
        self.grad = None
        self.grad_fn = None

    def zero_grad(self):
        self.grad = np.zeros_like(self.value)

    def backward(self, gradient=None):
        if gradient is None:
            if self.value.size == 1:
                gradient = np.ones_like(self.value)
            else:
                raise ValueError("Gradient required for vector-valued variable.")

        if self.grad is None:
            self.zero_grad()

        if len(gradient.shape) == len(self.value.shape) + 1:
            self.grad += gradient.sum(axis=0)
        elif len(gradient.shape) == len(self.value.shape):
            self.grad += gradient
        else:
            raise ValueError("Invalid gradient shape.")

        if self.grad_fn is not None:
            inputs_gradient = self.grad_fn(self, gradient)
            for (node, g) in zip(self.inputs, inputs_gradient):
                node.backward(g)
