import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from app.extensions import db
from app.models.job import Jobs, JobStatus
from app.models.model import Models
from app.controllers.model.reader import DatasetReader

from app.layers.activation import ReLU, Softmax
from app.layers.convolution import Conv2D, Flatten, MaxPool2D
from app.layers.layer import Layer
from app.layers.linear import Dense


class NeuralNetwork():
    def __init__(self, reader: DatasetReader, config):
        self.config = config
        self.reader = reader
        self.x_train, self.y_train = self.reader.get_training_dataset()
        self.x_test, self.y_test = self.reader.get_testing_dataset()
        self.model = None

    def create(self, model_name, layers_json):
        inputs = keras.Input(shape=self.reader.get_shape())
        x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
        for layer in layers_json:
            kerasLayer, activation = self.parseLayer(layer).getKerasLayer()
            x = kerasLayer(x)
            if activation is not None:
                x = activation(x)

        self.model = keras.Model(inputs=inputs, outputs=x)
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="acc")
            ],
        )

        model = Models(model_name)
        db.session.add(model)
        db.session.commit()

        models_path = self.config['MODELS_PATH']
        self.model.save(f'{models_path}/model_{model.model_id}')

    def load(self, model_id, job_id=None, from_checkpoint=False):
        models_path = self.config['MODELS_PATH']
        self.model = keras.models.load_model(f'{models_path}/model_{model_id}')

        if from_checkpoint:
            checkpoint_path = self.config['CHECKPOINTS_PATH']
            self.model.load_weights(f'{checkpoint_path}/model_{model_id}/checkpoint_{job_id}/checkpoint')

    def train(self, model_id: int, epochs: int, x_train=None, y_train=None):
        if self.model is None:
            print('No model has been loaded')
            return

        session = db.create_scoped_session()
        job = Jobs(status=JobStatus.IN_PROGRESS, model_id=model_id)
        session.add(job)
        session.commit()

        checkpoint_path = self.config['CHECKPOINTS_PATH']
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=f'{checkpoint_path}/model_{model_id}/checkpoint_{job.job_id}/checkpoint',
                save_weights_only=True
            )
        ]

        batch_size = 64
        if (x_train is not None) and (y_train is not None):
            history = self.model.fit(
                x_train, y_train,
                epochs=1, callbacks=callbacks
            )
        else:
            history = self.model.fit(
                self.x_train, self.y_train,
                batch_size=batch_size, epochs=epochs, callbacks=callbacks
            )

        val_dataset = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).batch(batch_size)
        evaluation_metrics = self.model.evaluate(val_dataset)

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

    def parseLayer(self, layer) -> Layer:
        print(layer)
        if (layer['__type__'] == Conv2D.__name__):
            return Conv2D(
                layer['filters'],
                layer['kernel_size'],
                layer['strides'],
                self.parseLayer(layer['activation'])
            )
        elif (layer['__type__'] == MaxPool2D.__name__):
            return MaxPool2D(layer['pool_size'], layer['strides'])
        elif (layer['__type__'] == Flatten.__name__):
            return Flatten()
        elif (layer['__type__'] == Dense.__name__):
            return Dense(layer['units'], self.parseLayer(layer['activation']))
        elif (layer['__type__'] == ReLU.__name__):
            return ReLU()
        elif (layer['__type__'] == Softmax.__name__):
            return Softmax()

        raise ValueError
