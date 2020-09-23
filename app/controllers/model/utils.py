from app.models.job import Jobs
from flask.app import Flask
import base64
from PIL import Image
from PIL import ImageOps
from PIL.ImageOps import invert
from io import BytesIO

from app.controllers.model.neural_network import NeuralNetwork
from app.controllers.model.reader import DatasetReader


def train_model(current_app: Flask, model_id: int, job_id: int, epochs: int, from_checkpoint: bool):
    with current_app.app_context():
        reader = DatasetReader(
            current_app.config['TRAINING_IMAGES'],
            current_app.config['TRAINING_LABELS'],
            current_app.config['TESTING_IMAGES'],
            current_app.config['TESTING_LABELS']
        )
        nn = NeuralNetwork(reader, current_app.config)
        nn.load(model_id, job_id=job_id, from_checkpoint=from_checkpoint)
        nn.train(model_id, epochs)


def serialize_checkpoint(job: Jobs):
    return {
        'checkpoint_id': job.job_id,
        'created_at': job.created_at
    }


def data_URL_to_number_array(data_URL):
    base64_img = data_URL.split(',', 1)[1]
    img = Image.open(BytesIO(base64.b64decode(base64_img))).convert('L')
    img = img.resize((20, 20))
    img = invert(img)
    img = ImageOps.autocontrast(img)
    img = ImageOps.expand(img, border=4)
    seq = list(img.getdata())
    return [
        [
            seq[i + 28 * j]
            for i in range(28)
        ] for j in range(28)
    ]
