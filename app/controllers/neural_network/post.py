from flask import request
from flask.json import jsonify
from app import app
from app.blueprints import neural_network
from app.controllers.neural_network.neural_network import NeuralNetwork
from app.controllers.neural_network.reader import DatasetReader


@neural_network.route('/predict', methods=['POST'])
def predict():
    reader = DatasetReader(
        app.config['TRAINING_IMAGES'],
        app.config['TRAINING_LABELS'],
        app.config['TESTING_IMAGES'],
        app.config['TESTING_LABELS']
    )
    nn = NeuralNetwork(reader)
    body = request.get_json()
    predictions, labels = nn.test(body.tests)

    return jsonify({
        predictions: predictions,
        labels: labels
    })
