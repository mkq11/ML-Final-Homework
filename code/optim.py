import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params = list(params)
    
    def step(self):
        raise NotImplementedError
    
    def zero_grad(self):
        for param in self.params:
            param.zero_grad()
    

class SGD(Optimizer):
    def __init__(self, params, lr=0.001, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.momentums = [np.zeros_like(param.value) for param in self.params]

    def step(self):
        for param, m in zip(self.params, self.momentums):
            m *= self.momentum
            m += param.grad
            param.value -= self.lr * (m + self.weight_decay * param.value) 
            