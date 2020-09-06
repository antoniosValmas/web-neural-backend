from io import BytesIO
from flask import Blueprint, request, current_app
from flask.json import jsonify
from threading import Thread
import base64
from PIL import Image
from PIL import ImageOps
from PIL.ImageOps import invert
from app.controllers.neural_network.neural_network import NeuralNetwork
from app.controllers.neural_network.reader import DatasetReader
from app.controllers.neural_network.utils import train_model

neural_network = Blueprint('neural_network', __name__, url_prefix='/neural-network')


@neural_network.route('/predict', methods=['POST'])
def predict():
    reader = DatasetReader(
        current_app.config['TRAINING_IMAGES'],
        current_app.config['TRAINING_LABELS'],
        current_app.config['TESTING_IMAGES'],
        current_app.config['TESTING_LABELS']
    )
    nn = NeuralNetwork(reader, current_app.config)
    body = request.get_json()
    url = body['url']
    base64_img = url.split(',', 1)[1]
    img = Image.open(BytesIO(base64.b64decode(base64_img))).convert('L')
    img = img.resize((28, 28))
    img = invert(img)
    img = ImageOps.autocontrast(img)
    seq = list(img.getdata())
    imageArray = [
        [
            seq[i + 28 * j]
            # if seq[i + 28 * j] > 50
            # else 0
            for i in range(28)
        ] for j in range(28)
    ]
    nn.load(from_checkpoint=True)
    predictions, labels = nn.test([imageArray])

    return jsonify({
        'predictions': predictions.tolist(),
        'labels': labels.tolist()
    })


@neural_network.route('/train', methods=['POST'])
def train():
    job = Thread(target=train_model, args=(current_app._get_current_object(),))
    job.start()

    return jsonify({
        'status': 'ok'
    })
