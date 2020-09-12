from app.models.model import Models
from flask import Blueprint, request, current_app
from flask.json import jsonify
from threading import Thread

from app.extensions import db
from app.models.job import Jobs, JobStatus
from app.controllers.model.neural_network import NeuralNetwork
from app.controllers.model.reader import DatasetReader
from app.controllers.model.utils import data_URL_to_number_array, serialize_checkpoint, train_model

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


@model.route('/<int:model_id>/checkpoints/<int:job_id>/predict', methods=['POST'])
def predict(model_id, job_id):
    reader = DatasetReader(
        current_app.config['TRAINING_IMAGES'],
        current_app.config['TRAINING_LABELS'],
        current_app.config['TESTING_IMAGES'],
        current_app.config['TESTING_LABELS']
    )
    nn = NeuralNetwork(reader, current_app.config)
    body = request.get_json()
    url: str = body['url']
    image_array = data_URL_to_number_array(url)
    nn.load(model_id, job_id, True)
    predictions, labels = nn.test([image_array])

    return jsonify({
        'predictions': predictions.tolist(),
        'labels': labels.tolist()
    })


@model.route('/<int:model_id>/train', methods=['POST'])
def train(model_id):
    body = request.get_json()
    epochs: int = body['epochs']
    job_id: int = body['checkpoint_id']
    from_checkpoint: bool = body['fromCheckpoint']
    session = db.create_scoped_session()
    running_job = session.query(Jobs).filter_by(status=JobStatus.IN_PROGRESS).first()
    if running_job:
        return jsonify({
            'status': 'JOB_ALREADY_RUNNING'
        })
    print(job_id)

    job = Thread(
        target=train_model,
        args=(current_app._get_current_object(), model_id, job_id, epochs, from_checkpoint)
    )
    job.start()

    return jsonify({
        'status': 'JOB_STARTED'
    })


@model.route('/<int:model_id>/checkpoints/<int:job_id>/train/user-input', methods=["POST"])
def train_user_input(model_id, job_id):
    reader = DatasetReader(
        current_app.config['TRAINING_IMAGES'],
        current_app.config['TRAINING_LABELS'],
        current_app.config['TESTING_IMAGES'],
        current_app.config['TESTING_LABELS']
    )
    nn = NeuralNetwork(reader, current_app.config)
    body = request.get_json()
    images = [
        data_URL_to_number_array(image['imageURL'])
        for image in body
    ]
    labels = [image['label'] for image in body]
    nn.load(model_id, job_id, True)
    nn.train(model_id, 1, images, labels)
    return jsonify({
        'status': 'ok'
    })


@model.route('/checkpoints', methods=['GET'])
def get_checkpoints():
    session = db.create_scoped_session()
    models = session.query(Models).all()

    return jsonify([
        {
            'model_id': model.model_id,
            'model_name': model.model_name,
            'checkpoints': [
                serialize_checkpoint(job)
                for job in model.jobs
            ]
        }
        for model in models
    ])
