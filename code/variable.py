from typing import Union
import numpy as np

import autograd


def broadcast_if_needed(x, y):
    y = y if isinstance(y, Variable) else Variable(y)
    x_shape = x.value.shape
    y_shape = y.value.shape
    if x_shape != y_shape:
        broadcast_shape = np.broadcast_shapes(x_shape, y_shape)
        return (
            Variable(autograd.BroadcastNode(x, broadcast_shape)),
            Variable(autograd.BroadcastNode(y, broadcast_shape)),
        )
    else:
        return x, y


class Variable:
    def __init__(self, x: Union[np.ndarray, autograd.CalculationNode]):

        self.grad_fn = None
        self.grad = None
        self.requires_grad = False
        self._calc_node = None
        self._ref_count = 0
        self._current_grad = None

        if isinstance(x, autograd.CalculationNode):
            self.value = x.forward()
            self.grad_fn = x.backward
            self._calc_node = x
        elif isinstance(x, np.ndarray):
            self.value = np.array(x, dtype=np.float32)
        elif isinstance(x, Variable):
            self.value = x.value
        elif isinstance(x, float) or isinstance(x, int):
            self.value = np.array([x], dtype=np.float32)
        else:
            raise ValueError("Invalid input.")

    def _update_ref_count(self):
        self._ref_count = 0
        if self._calc_node is None:
            return
        for var in self._calc_node.inputs:
            if not isinstance(var, Variable):
                continue
            var._update_ref_count()
            var._ref_count += 1

    def zero_grad(self):
        if not self.requires_grad:
            raise ValueError(
                "Cannot zero gradient of a variable that requires no gradient."
            )
        self.grad = np.zeros_like(self.value)

    def backward(self, gradient=None):
        if gradient is None:
            if self.value.size == 1:
                grad = np.ones_like(self.value)
            else:
                raise ValueError("Gradient required for vector-valued variable.")
        else:
            grad = gradient
        self._update_ref_count()
        self._backward_impl(grad)

    def _backward_impl(self, gradient):
        if self._current_grad is None:
            self._current_grad = gradient.copy()

        if self._ref_count != 0:
            self._ref_count -= 1
            self._current_grad += gradient
            return

        if self.requires_grad:
            if self.grad is None:
                self.zero_grad()

            if len(self._current_grad.shape) == len(self.value.shape) + 1:
                self.grad += self._current_grad.sum(axis=0)
            elif len(self._current_grad.shape) == len(self.value.shape):
                self.grad += self._current_grad
            else:
                raise ValueError("Invalid gradient shape.")

        if self.grad_fn is not None:
            inputs_gradient = self.grad_fn(self._current_grad)
            for (node, g) in inputs_gradient:
                node.backward(g)

        self._current_grad = None

    def __add__(self, other):
        x, y = broadcast_if_needed(self, other)
        return Variable(autograd.AddNode(x, y))

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        x, y = broadcast_if_needed(self, other)
        return Variable(autograd.SubNode(x, y))

    def __rsub__(self, other):
        return self - other

    def __mul__(self, other):
        x, y = broadcast_if_needed(self, other)
        return Variable(autograd.MulNode(x, y))

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        return Variable(autograd.MatMulNode(self, other))

    def __rmatmul__(self, other):
        return self @ other

    def __truediv__(self, other):
        x, y = broadcast_if_needed(self, other)
        return Variable(autograd.DivNode(x, y))

    def __rtruediv__(self, other):
        return self / other

    def __pow__(self, other: float):
        return Variable(autograd.PowNode(self, other))

    def mean(self, axis=None):
        return Variable(autograd.MeanNode(self, axis))

    def var(self, axis=None):
        err = self - self.mean(axis)
        sq_err = err ** 2
        return sq_err.mean(axis)  # * size / (size - 1)
