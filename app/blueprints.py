from flask import Blueprint

users = Blueprint('users', __name__, url_prefix='users')
neural_network = Blueprint('neural_network', __name__, url_prefix='neural-network')

blueprints = [users, neural_network]