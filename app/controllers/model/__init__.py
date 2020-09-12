from io import BytesIO
from flask import Blueprint, request, current_app
from flask.json import jsonify
from threading import Thread
import base64
from PIL import Image
from PIL import ImageOps
from PIL.ImageOps import invert

from app.extensions import db
from app.models.job import Jobs, JobStatus
from app.controllers.model.neural_network import NeuralNetwork
from app.controllers.model.reader import DatasetReader
from app.controllers.model.utils import train_model

model = Blueprint('models', __name__, url_prefix='/models')


@model.route('/', methods=['POST'])
def create_model():
    reader = DatasetReader(
        current_app.config['TRAINING_IMAGES'],
        current_app.config['TRAINING_LABELS'],
        current_app.config['TESTING_IMAGES'],
        current_app.config['TESTING_LABELS']
    )
    nn = NeuralNetwork(reader, current_app.config)
    body = request.get_json()
    nn.create(body['model_name'], body['model_layers'])

    return jsonify({
        'status': 'ok'
    })


@model.route('/<model_id>/predict', methods=['POST'])
def predict(model_id):
    reader = DatasetReader(
        current_app.config['TRAINING_IMAGES'],
        current_app.config['TRAINING_LABELS'],
        current_app.config['TESTING_IMAGES'],
        current_app.config['TESTING_LABELS']
    )
    nn = NeuralNetwork(reader, current_app.config)
    body = request.get_json()
    url: str = body['url']
    base64_img = url.split(',', 1)[1]
    img = Image.open(BytesIO(base64.b64decode(base64_img))).convert('L')
    img = img.resize((28, 28))
    img = invert(img)
    img = ImageOps.autocontrast(img)
    seq = list(img.getdata())
    imageArray = [
        [
            seq[i + 28 * j]
            for i in range(28)
        ] for j in range(28)
    ]
    nn.load(model_id, True)
    predictions, labels = nn.test([imageArray])

    return jsonify({
        'predictions': predictions.tolist(),
        'labels': labels.tolist()
    })


@model.route('/<int:model_id>/train', methods=['POST'])
def train(model_id):
    body = request.get_json()
    epochs: int = body['epochs']
    from_checkpoint: bool = body['fromCheckpoint']
    session = db.create_scoped_session()
    running_job = session.query(Jobs).filter_by(status=JobStatus.IN_PROGRESS).first()
    if running_job:
        return jsonify({
            'status': 'JOB_ALREADY_RUNNING'
        })

    job = Thread(target=train_model, args=(current_app._get_current_object(), model_id, epochs, from_checkpoint))
    job.start()

    return jsonify({
        'status': 'JOB_STARTED'
    })
