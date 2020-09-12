from tensorflow.keras import layers
from app.layers.layer import Layer


class ReLU(Layer):
    def getKerasLayer(self):
        return layers.ReLU(), None


class Softmax(Layer):
    def getKerasLayer(self):
        return layers.Softmax(), None
