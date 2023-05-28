import numpy as np
from variable import Variable


def add_backward(node, gradient):
    return (gradient, gradient)


def add(x, y):
    z = Variable(x.value + y.value)
    z.inputs = [x, y]
    z.grad_fn = add_backward
    return z


def mul_backward(node, gradient):
    g1 = gradient.dot(node.inputs[1].value.T)
    g2 = node.inputs[0].value.T.dot(gradient)
    return (g1, g2)


def mul(x, y):
    z = Variable(x.value @ y.value)
    z.inputs = [x, y]
    z.grad_fn = mul_backward
    return z


def relu_backward(node, gradient):
    return (gradient * (node.inputs[0].value > 0),)


def relu(x):
    out = Variable(x.value * (x.value > 0))
    out.inputs = [x]
    out.grad_fn = relu_backward
    return out


def mse_loss_backward(node, gradient):
    return (
        gradient
        * 2
        * (node.inputs[0].value - node.inputs[1].value)
        / node.inputs[0].value.size,
        gradient
        * 2
        * (node.inputs[1].value - node.inputs[0].value)
        / node.inputs[1].value.size,
    )


def mse_loss(x, y):
    z = Variable(np.square(x.value - y.value).mean())
    z.inputs = [x, y]
    z.grad_fn = mse_loss_backward
    return z


def sum_backward(node, gradient):
    return (gradient * np.ones_like(node.inputs[0].value),)


def sum(x):
    z = Variable(np.sum(x.value))
    z.inputs = [x]
    z.grad_fn = sum_backward
    return z
