from app.extensions import db
from app.models.base import Base
from enum import Enum


class JobStatus(Enum):
    IN_PROGRESS = 'IN_PROGRESS'
    FINISHED = 'FINISHED'


class Jobs(Base):
    __tablename__ = 'jobs'

    def __init__(self, status):
        self.status = status

    job_id = db.Column(db.Integer, primary_key=True)
    status = db.Column(db.Enum(JobStatus))
    history_loss = db.Column(db.JSON(), nullable=True)
    history_acc = db.Column(db.JSON(), nullable=True)
    evaluation_loss = db.Column(db.Float, nullable=True)
    evaluation_acc = db.Column(db.Float, nullable=True)
