from fastapi import FastAPI, HTTPException, Depends
from typing import Dict, Any
from pydantic import BaseModel
import pandas as pd
import os, pickle
from work.ml.data import process_data

savepath = 'model'
filename = ['trained_model.pkl', 'encoder.pkl', 'labelizer.pkl']

class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
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

def get_model():
    model_path = os.path.join(savepath, filename[0])
    encoder_path = os.path.join(savepath, filename[1])
    lb_path = os.path.join(savepath, filename[2])
    if os.path.isfile(model_path):
        model = pickle.load(open(model_path, "rb"))
        encoder = pickle.load(open(encoder_path, "rb"))
        lb = pickle.load(open(lb_path, "rb"))
        return model, encoder, lb
    else:
        raise HTTPException(status_code=500, detail="Model not found")

app = FastAPI(
    title="Inference API",
    description="An API that takes a sample and runs an inference",
    version="1.0.0"
)

@app.get("/")
async def greetings():
    return "Welcome to our model API"

def transform_data(data: Dict[str, Any], encoder, lb):
    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]
    sample = pd.DataFrame(data, index=[0])
    sample, _, _, _ = process_data(
        sample,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb
    )
    return sample

def make_prediction(sample: pd.DataFrame, model):
    prediction = model.predict(sample)
    return '>50K' if prediction[0] > 0.5 else '<=50K'


@app.post("/inference/")
async def ingest_data(inference: InputData, model_encoder_lb: tuple = Depends(get_model)):
    model, encoder, lb = model_encoder_lb
    data = inference.dict()
    sample = transform_data(data, encoder, lb)
    prediction = make_prediction(sample, model)
    data['prediction'] = prediction
    return data
