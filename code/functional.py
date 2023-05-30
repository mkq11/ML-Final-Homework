from variable import Variable

import autograd

def relu(x):
    return Variable(autograd.ReLUNode(x))

def cross_entropy_loss(x, y):
    return Variable(autograd.CrossEntropyLossNode(x, y))