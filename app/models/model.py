from app.extensions import db
from app.models.base import Base


class Models(Base):
    __tablename__ = 'models'

    def __init__(self, model_name):
        self.model_name = model_name

    model_id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(200), nullable=False)
