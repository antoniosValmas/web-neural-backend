import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from app.controllers.neural_network.reader import DatasetReader


class NeuralNetwork():
    def __init__(self, reader: DatasetReader):
        self.reader = reader
        self.x_train, self.y_train = self.reader.get_training_dataset()
        self.x_test, self.y_test = self.reader.get_testing_dataset()
        self.model = None

    def load(self, from_checkpoint=False):
        inputs = keras.Input(shape=self.reader.get_shape())
        x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
        x = layers.Conv2D(filters=32, kernel_size=(5, 5), activation="relu")(x)
        x = layers.MaxPool2D(pool_size=(5, 5), strides=(1, 1))(x)
        x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu")(x)
        x = layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(units=100, activation="relu")(x)
        outputs = layers.Dense(self.reader.get_number_of_classes(), activation="softmax")(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="acc")
            ],
        )

        if from_checkpoint:
            self.model.load_weights('./models/checkpoint')

    def train(self):
        if self.model is None:
            print('No model has been loaded')
            return

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath='./models/checkpoint',
                save_weights_only=True
            )
        ]

        batch_size = 64
        history = self.model.fit(
            self.x_train, self.y_train,
            batch_size=batch_size, epochs=3, callbacks=callbacks
        )

        val_dataset = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).batch(batch_size)
        evaluation_metrics = self.model.evaluate(val_dataset)

        return history, evaluation_metrics

    def test(self, data):
        predictions = self.model.predict(data)
        labels = np.zeros(len(predictions))

        for i, prediction in enumerate(predictions):
            labels[i] = max(range(len(prediction)), key=prediction.__getitem__)

        return predictions, labels
