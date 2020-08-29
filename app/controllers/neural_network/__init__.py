from flask import Blueprint, request, current_app
from flask.json import jsonify
from app.controllers.neural_network.neural_network import NeuralNetwork
from app.controllers.neural_network.reader import DatasetReader


neural_network = Blueprint('neural_network', __name__, url_prefix='/neural-network')


@neural_network.route('/predict', methods=['POST'])
def predict():
    reader = DatasetReader(
        current_app.config['TRAINING_IMAGES'],
        current_app.config['TRAINING_LABELS'],
        current_app.config['TESTING_IMAGES'],
        current_app.config['TESTING_LABELS']
    )
    nn = NeuralNetwork(reader)
    body = request.get_json()
    nn.load()
    print(body['tests'])
    predictions, labels = nn.test(body['tests'])

    return jsonify({
        'predictions': predictions.tolist(),
        'labels': labels.tolist()
    })
