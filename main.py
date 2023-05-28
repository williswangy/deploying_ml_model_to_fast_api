from fastapi import FastAPI, HTTPException
from typing import Union, Optional
from pydantic import BaseModel
import pandas as pd
import os, pickle
from work.ml.data import process_data

# path to saved artifacts
SAVEPATH = 'model'
FILENAME = ['trained_model.pkl', 'encoder.pkl', 'labelizer.pkl']

# Instantiate FastAPI app
app = FastAPI(title="Inference API",
              description="An API that takes a sample and runs an inference",
              version="1.0.0")

# Class for defining the input data model
class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: strls
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
                'age': 50,
                'workclass': "Private",
                'fnlgt': 234721,
                'education': "Doctorate",
                'education_num': 16,
                'marital_status': "Separated",
                'occupation': "Exec-managerial",
                'relationship': "Not-in-family",
                'race': "Black",
                'sex': "Female",
                'capital_gain': 0,
                'capital_loss': 0,
                'hours_per_week': 50,
                'native_country': "United-States"
            }
        }

# Global vars for model artifacts
model = None
encoder = None
lb = None


@app.on_event("startup")
async def load_model_artifacts():
    global model, encoder, lb
    try:
        model = pickle.load(open(os.path.join(SAVEPATH, FILENAME[0]), "rb"))
        encoder = pickle.load(open(os.path.join(SAVEPATH, FILENAME[1]), "rb"))
        lb = pickle.load(open(os.path.join(SAVEPATH, FILENAME[2]), "rb"))
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model artifacts not found")


@app.get("/")
async def greetings():
    return "Welcome to our model API"


def get_prediction(sample):
    prediction = model.predict(sample)
    return '>50K' if prediction[0] > 0.5 else '<=50K'


def prepare_data_for_prediction(inference: InputData):
    data = inference.dict()
    sample = pd.DataFrame(data, index=[0])
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    sample, _, _, _ = process_data(
        sample,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb
    )
    return sample, data


@app.post("/inference/")
async def ingest_data(inference: InputData):
    sample, data = prepare_data_for_prediction(inference)
    data['prediction'] = get_prediction(sample)
    return data
