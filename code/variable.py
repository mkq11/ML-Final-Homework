from typing import Union
import numpy as np

import autograd

class Variable:
    def __init__(self, x : Union[np.ndarray, autograd.CalculationNode]):
        
        self.grad_fn = None
        self.grad = None
        self.requires_grad = False

        if isinstance(x, autograd.CalculationNode):
            self.value = x.forward()
            self.grad_fn = x.backward
        elif isinstance(x, np.ndarray):
            self.value = np.array(x, dtype=np.float32)
        elif isinstance(x, Variable):
            self.value = x.value
        elif isinstance(x, float) or isinstance(x, int):
            self.value = np.array(x, dtype=np.float32)
        else:
            raise ValueError("Invalid input.")

    def zero_grad(self):
        if not self.requires_grad:
            raise ValueError(
                "Cannot zero gradient of a variable that requires no gradient."
            )
        self.grad = np.zeros_like(self.value)

    def backward(self, gradient=None):
        if gradient is None:
            if self.value.size == 1:
                gradient = np.ones_like(self.value)
            else:
                raise ValueError("Gradient required for vector-valued variable.")

        if self.requires_grad:
            if self.grad is None:
                self.zero_grad()

            if len(gradient.shape) == len(self.value.shape) + 1:
                self.grad += gradient.sum(axis=0)
            elif len(gradient.shape) == len(self.value.shape):
                self.grad += gradient
            else:
                raise ValueError("Invalid gradient shape.")

        if self.grad_fn is not None:
            inputs_gradient = self.grad_fn(gradient)
            for (node, g) in inputs_gradient:
                node.backward(g)

    def __add__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        return Variable(autograd.AddNode(self, other))
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        return Variable(autograd.SubNode(self, other))
    
    def __rsub__(self, other):
        return self - other
    
    def __mul__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        return Variable(autograd.MulNode(self, other))
    
    def __rmul__(self, other):
        return self * other
    
    def __matmul__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        return Variable(autograd.MatMulNode(self, other))
    
    def __rmatmul__(self, other):
        return self @ other
    
    def __truediv__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        return Variable(autograd.DivNode(self, other))
    
    def __rtruediv__(self, other):
        return self / other
    
