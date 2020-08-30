from flask import Flask

from app.models import models
from app.extensions import db
from app.controllers import blueprints

app = Flask(__name__)
app.config.from_object('config')

db.init_app(app)

for blueprint in blueprints:
    app.register_blueprint(blueprint)

for model in models:
    app.logger.info(f'Model registered: {model.__name__}')
