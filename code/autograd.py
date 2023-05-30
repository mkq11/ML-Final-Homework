import numpy as np


class CalculationNode:
    def __init__(self, *args):
        self.inputs = args

    def forward(self):
        raise NotImplementedError

    def backward(self, gradient):
        raise NotImplementedError


class AddNode(CalculationNode):
    def __init__(self, x, y):
        super().__init__(x, y)

    def forward(self):
        return self.inputs[0].value + self.inputs[1].value

    def backward(self, gradient):
        return zip(self.inputs, (gradient, gradient))


class SubNode(CalculationNode):
    def __init__(self, x, y):
        super().__init__(x, y)

    def forward(self):
        return self.inputs[0].value - self.inputs[1].value

    def backward(self, gradient):
        return zip(self.inputs, (gradient, -gradient))


class MulNode(CalculationNode):
    def __init__(self, x, y):
        super().__init__(x, y)

    def forward(self):
        return self.inputs[0].value * self.inputs[1].value

    def backward(self, gradient):
        return zip(
            self.inputs,
            (gradient * self.inputs[1].value, gradient * self.inputs[0].value),
        )


class DivNode(CalculationNode):
    def __init__(self, x, y):
        super().__init__(x, y)

    def forward(self):
        return self.inputs[0].value / self.inputs[1].value

    def backward(self, gradient):
        return zip(
            self.inputs,
            (
                gradient / self.inputs[1].value,
                -gradient * self.inputs[0].value / self.inputs[1].value ** 2,
            ),
        )


class MatMulNode(CalculationNode):
    def __init__(self, x, y):
        super().__init__(x, y)

    def forward(self):
        return self.inputs[0].value @ self.inputs[1].value

    def backward(self, gradient):
        return zip(
            self.inputs,
            (
                gradient @ self.inputs[1].value.T,
                self.inputs[0].value.T @ gradient,
            ),
        )


class ReLUNode(CalculationNode):
    def __init__(self, x):
        super().__init__(x)

    def forward(self):
        return self.inputs[0].value * (self.inputs[0].value > 0)

    def backward(self, gradient):
        return ((self.inputs[0], gradient * (self.inputs[0].value > 0)),)


class CrossEntropyLossNode(CalculationNode):
    def __init__(self, x, y):
        super().__init__(x, y)

    def _softmax(self):
        x = self.inputs[0].value
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True) + 1e-5

    def forward(self):
        y = self.inputs[1].value
        softmax = self._softmax()
        return -np.sum(y * np.log(softmax)) / softmax.shape[0]

    def backward(self, gradient):
        y = self.inputs[1].value
        softmax = self._softmax()
        return ((self.inputs[0], gradient * (softmax - y) / softmax.shape[0]),)
