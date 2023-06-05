from variable import Variable

import autograd


def relu(x):
    return Variable(autograd.ReLUNode(x))


def cross_entropy_loss(x, y):
    return Variable(autograd.CrossEntropyLossNode(x, y))


def mse_loss(x, y):
    return Variable(autograd.MSELossNode(x, y))


def conv2d(x, w, b, stride=1, padding=0):
    return Variable(autograd.Conv2dNode(x, w, b, stride, padding))
