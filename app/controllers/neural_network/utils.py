from app.controllers.neural_network.neural_network import NeuralNetwork
from app.controllers.neural_network.reader import DatasetReader


def train_model(current_app):
    with current_app.app_context():
        reader = DatasetReader(
            current_app.config['TRAINING_IMAGES'],
            current_app.config['TRAINING_LABELS'],
            current_app.config['TESTING_IMAGES'],
            current_app.config['TESTING_LABELS']
        )
        nn = NeuralNetwork(reader, current_app.config)
        nn.load(from_checkpoint=True)
        nn.train()
