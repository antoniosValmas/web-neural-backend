from typing import Tuple
from tensorflow.keras import layers
from app.layers.layer import Layer


class Conv2D(Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: Tuple[int, int],
        strides: Tuple[int, int],
        activation: Layer
    ):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation

    def getKerasLayer(self):
        return (
            layers.Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=self.strides
            ),
            self.activation.getKerasLayer()[0]
        )


class MaxPool2D(Layer):
    def __init__(
        self,
        pool_size: Tuple[int, int],
        strides: Tuple[int, int],
    ):
        self.pool_size = pool_size
        self.strides = strides

    def getKerasLayer(self):
        return (
            layers.MaxPool2D(
                pool_size=self.pool_size,
                strides=self.strides
            ), None)


class Flatten(Layer):
    def getKerasLayer(self):
        return layers.Flatten(), None
