import sqlalchemy as db
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
import pandas as pd
import numpy as np
from sqlalchemy.orm import sessionmaker
import MetaFunctions as MF
import LandmarkMetaFunctions as LMF
import base64


engine = create_engine('sqlite:///DB\\MetaFeatures.db')
connection = engine.connect()
metadata = db.MetaData()
Base = declarative_base()

def select_all():
    Session = sessionmaker(bind=engine)
    session = Session()
    meta_features = np.empty(0)
    for row in session.query(MetaFeatures):
        meta_features = np.append(meta_features, row)
    return meta_features

class MetaFeatures(Base):
    __tablename__ = 'MetaFeatures'
    id = db.Column(db.Integer, primary_key = True)
    hash_id = db.Column('HashID', db.String(900))
    name = db.Column('Name', db.String(255))
    meta_features = db.Column('MetaFeatures', db.String(3000))
    l_meta_features = db.Column('LMetaFeatures', db.String(3000))
    hyperparameters = db.Column('Hyperparameters', db.String(3000))
    fsa = db.Column('FSA', db.String(255))


    def __init__(self, data, name, hyper_params, fsa, to_string_bool):
        self.data = data
        self.name = name
        self.hash_id = base64.b64encode(name.encode('utf-8'))
        self.hyperparameters = hyper_params
        self.fsa = fsa
        self.meta_features = MF.compute_meta_features(data, to_string_bool)
        self.l_meta_features= LMF.compute_l_meta_features(data, to_string_bool)


