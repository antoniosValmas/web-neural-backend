from tensorflow.keras import layers
from app.layers.layer import Layer


class Dense(Layer):
    def __init__(self, units: int, activation: Layer):
        self.units = units
        self.activation = activation

    def getKerasLayer(self):
        return (
            layers.Dense(units=self.units),
            self.activation.getKerasLayer()[0]
        )
