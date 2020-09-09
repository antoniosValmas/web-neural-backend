import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from app.extensions import db
from app.models.job import Jobs, JobStatus
from app.controllers.neural_network.reader import DatasetReader


class NeuralNetwork():
    def __init__(self, reader: DatasetReader, config):
        self.config = config
        self.reader = reader
        self.x_train, self.y_train = self.reader.get_training_dataset()
        self.x_test, self.y_test = self.reader.get_testing_dataset()
        self.model = None

    def load(self, from_checkpoint=False):
        inputs = keras.Input(shape=self.reader.get_shape())
        x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
        x = layers.experimental.preprocessing.RandomContrast(0.5)(x)
        x = layers.Conv2D(filters=32, kernel_size=(5, 5), activation="relu")(x)
        x = layers.Conv2D(filters=32, kernel_size=(5, 5), activation="relu")(x)
        x = layers.MaxPool2D(pool_size=(5, 5), strides=(1, 1))(x)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(x)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(x)
        x = layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1))(x)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(x)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(x)
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
            checkpoint_path = self.config['CHECKPOINTS_PATH']
            self.model.load_weights(f'{checkpoint_path}/checkpoint')

    def train(self, epochs: int):
        if self.model is None:
            print('No model has been loaded')
            return

        session = db.create_scoped_session()
        job = Jobs(status=JobStatus.IN_PROGRESS)
        session.add(job)
        session.commit()

        checkpoint_path = self.config['CHECKPOINTS_PATH']
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=f'{checkpoint_path}/checkpoint',
                save_weights_only=True
            )
        ]

        batch_size = 64
        history = self.model.fit(
            self.x_train, self.y_train,
            batch_size=batch_size, epochs=epochs, callbacks=callbacks
        )

        val_dataset = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).batch(batch_size)
        evaluation_metrics = self.model.evaluate(val_dataset)
        # predictions = self.model.predict(self.x_test)
        # for i, prediction in enumerate(predictions):
        #     max_index = max(range(len(prediction)), key=prediction.__getitem__)
        #     if self.y_test[i] != max_index:
        #         print(f'Expected {self.y_test[i]} predicted {max_index}')
        #         self.reader.display(self.x_test[i])

        session = db.create_scoped_session()
        running_job = session.query(Jobs).filter_by(status=JobStatus.IN_PROGRESS).first()
        running_job.status = JobStatus.FINISHED
        running_job.history_loss = history.history['loss']
        running_job.history_acc = history.history['acc']
        running_job.evaluation_loss = evaluation_metrics[0]
        running_job.evaluation_acc = evaluation_metrics[1]
        session.commit()

    def test(self, data):
        predictions = self.model.predict(data)
        labels = np.zeros(len(predictions))

        for i, prediction in enumerate(predictions):
            labels[i] = max(range(len(prediction)), key=prediction.__getitem__)

        return predictions, labels
