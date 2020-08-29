from flask import Flask
from app.blueprints import blueprints

app = Flask(__name__)
app.config.from_object('config')
for blueprint in blueprints:
    app.register_blueprint(blueprint)
