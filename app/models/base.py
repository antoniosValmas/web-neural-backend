from app.extensions import db
import datetime


class Base(db.Model):
    """Base model that provides some common features, such as automatic 'created'
    and 'modified' date columns.
    """

    __abstract__ = True
    created_at = db.Column(db.DateTime, server_default=db.func.now(), nullable=False)
    updated_at = db.Column(db.DateTime, server_default=db.func.now(),
                           onupdate=datetime.datetime.now, nullable=False)
