import numpy as np


class CalculationNode:
    def __init__(self, *args):
        self.inputs = args

    def forward(self):
        raise NotImplementedError

    def backward(self, gradient):
        raise NotImplementedError


class BroadcastNode(CalculationNode):
    def __init__(self, x, shape):
        super().__init__(x)
        self.shape = shape
        x_shape = np.ones(len(shape), dtype=np.int)
        x_shape[-len(x.value.shape) :] = np.array(x.value.shape)
        self.broadcast_axis = np.where(x_shape != self.shape)[0]
        self.broadcast_axis = tuple(self.broadcast_axis)

    def forward(self):
        return np.broadcast_to(self.inputs[0].value, self.shape)

    def backward(self, gradient):
        return ((self.inputs[0], gradient.sum(axis=self.broadcast_axis)),)


class AddNode(CalculationNode):
    def __init__(self, x, y):
        if x.value.shape != y.value.shape:
            raise NotImplementedError
        super().__init__(x, y)

    def forward(self):
        return self.inputs[0].value + self.inputs[1].value

    def backward(self, gradient):
        return zip(self.inputs, (gradient, gradient))


class SubNode(CalculationNode):
    def __init__(self, x, y):
        if x.value.shape != y.value.shape:
            raise NotImplementedError
        super().__init__(x, y)

    def forward(self):
        return self.inputs[0].value - self.inputs[1].value

    def backward(self, gradient):
        return zip(self.inputs, (gradient, -gradient))


class MulNode(CalculationNode):
    def __init__(self, x, y):
        if x.value.shape != y.value.shape:
            raise NotImplementedError
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
        if x.value.shape != y.value.shape:
            raise NotImplementedError
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


class PowNode(CalculationNode):
    def __init__(self, x, y):
        super().__init__(x, y)

    def forward(self):
        return self.inputs[0].value ** self.inputs[1]

    def backward(self, gradient):
        return (
            (
                self.inputs[0],
                gradient
                * self.inputs[1]
                * self.inputs[0].value ** (self.inputs[1] - 1),
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


class MeanNode(CalculationNode):
    def __init__(self, x, axis=None):
        super().__init__(x)
        self.in_shape = np.array(x.value.shape)
        self.axis = axis if axis is not None else tuple(range(len(self.in_shape)))
        self.keep_shape = self.in_shape.copy()
        self.keep_shape[axis] = 1

    def forward(self):
        return np.mean(self.inputs[0].value, axis=self.axis)

    def backward(self, gradient):
        grad = np.broadcast_to(gradient.reshape(self.keep_shape), self.in_shape).copy()
        grad /= np.prod(self.in_shape[self.axis])
        return ((self.inputs[0], grad),)
