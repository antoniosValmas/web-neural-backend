from flask import Flask
# from flask_cors import CORS
from app.controllers import blueprints

app = Flask(__name__)
app.config.from_object('config')

# CORS(app)

for blueprint in blueprints:
    app.register_blueprint(blueprint)
