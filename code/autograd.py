import numpy as np


class CalculationNode:
    def __init__(self, *args):
        self.inputs = args

    def forward(self):
        raise NotImplementedError

    def backward(self, gradient):
        raise NotImplementedError


class ReshapeNode(CalculationNode):
    def __init__(self, x, shape):
        super().__init__(x)
        self.shape = shape

    def forward(self):
        return self.inputs[0].value.reshape(self.shape)

    def backward(self, gradient):
        return ((self.inputs[0], gradient.reshape(self.inputs[0].value.shape)),)


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
        grad = gradient.sum(axis=self.broadcast_axis)
        grad = grad.reshape(self.inputs[0].value.shape)
        return ((self.inputs[0], grad),)


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


class MSELossNode(CalculationNode):
    def __init__(self, x, y):
        super().__init__(x, y)

    def forward(self):
        return np.mean((self.inputs[0].value - self.inputs[1].value) ** 2)

    def backward(self, gradient):
        return (
            (self.inputs[0], gradient * 2 * (self.inputs[0].value - self.inputs[1].value)),
            (self.inputs[1], gradient * 2 * (self.inputs[1].value - self.inputs[0].value)),
        )

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
    def __init__(self, x, axis=None, keepdims=False):
        super().__init__(x)
        self.in_shape = np.array(x.value.shape)
        self.axis = axis if axis is not None else tuple(range(len(self.in_shape)))
        self.keep_shape = self.in_shape.copy()
        self.keep_shape[axis, ] = 1
        self.keepdims = keepdims

    def forward(self):
        return np.mean(self.inputs[0].value, axis=self.axis, keepdims=self.keepdims)

    def backward(self, gradient):
        grad = np.broadcast_to(gradient.reshape(self.keep_shape), self.in_shape).copy()
        grad /= np.prod(self.in_shape[self.axis, ])
        return ((self.inputs[0], grad),)


class Conv2dNode(CalculationNode):
    def __init__(self, x, w, b, stride=1, padding=0):
        super().__init__(x, w, b)
        self.stride = stride
        self.padding = padding
        self.x_shape = x.value.shape
        self.w_shape = w.value.shape
        self.b_shape = b.value.shape
        self.x_pad = np.pad(
            x.value,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            "constant",
            constant_values=0,
        )

    def forward(self):
        x = self.x_pad
        w = self.inputs[1].value
        b = self.inputs[2].value
        n, c, xh, xw = x.shape
        f, c, fh, fw = w.shape
        oh = (xh - fh) // self.stride + 1
        ow = (xw - fw) // self.stride + 1
        out = np.zeros((n, f, oh, ow))
        for i in range(oh):
            for j in range(ow):
                out[:, :, i, j] = np.sum(
                    x[
                        :,
                        :,
                        i * self.stride : i * self.stride + fh,
                        j * self.stride : j * self.stride + fw,
                    ].reshape(n, 1, c, fh, fw)
                    * w,
                    axis=(2, 3, 4),
                )
        return out + b.reshape(1, f, 1, 1)

    def backward(self, gradient):
        x = self.x_pad
        w = self.inputs[1].value
        b = self.inputs[2].value
        n, c, xh, xw = x.shape
        f, c, fh, fw = w.shape
        oh = (xh - fh) // self.stride + 1
        ow = (xw - fw) // self.stride + 1
        dx = np.zeros_like(x)
        dw = np.zeros_like(w)
        db = np.zeros_like(b)
        for i in range(oh):
            for j in range(ow):
                dx[
                    :,
                    :,
                    i * self.stride : i * self.stride + fh,
                    j * self.stride : j * self.stride + fw,
                ] += np.sum(
                    gradient[:, :, i : i + 1, j : j + 1].reshape(n, f, 1, 1, 1)
                    * w.reshape(1, f, c, fh, fw),
                    axis=1,
                )
                dw += np.sum(
                    x[
                        :,
                        :,
                        i * self.stride : i * self.stride + fh,
                        j * self.stride : j * self.stride + fw,
                    ].reshape(n, 1, c, fh, fw)
                    * gradient[:, :, i : i + 1, j : j + 1].reshape(n, f, 1, 1, 1),
                    axis=0,
                )
        db += np.sum(gradient, axis=(0, 2, 3))
        ub = -self.padding if self.padding > 0 else None
        dx = dx[:, :, self.padding : ub, self.padding : ub]
        return ((self.inputs[0], dx), (self.inputs[1], dw), (self.inputs[2], db))
