import pytest, os, logging, pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError

from starter.work.ml.model import inference, compute_model_metrics, compute_confusion_matrix
from starter.work.ml.data import process_data

DATA_PATH = "./data/census.csv"
MODEL_PATH = "./model/trained_model.pkl"
CAT_FEATURES = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

@pytest.fixture(scope="module")
def data():
    try:
        return pd.read_csv(DATA_PATH)
    except FileNotFoundError as err:
        logging.error("File not found")
        raise err

@pytest.fixture(scope="module")
def model():
    if os.path.isfile(MODEL_PATH):
        try:
            return pickle.load(open(MODEL_PATH, 'rb'))
        except Exception as err:
            logging.error("Testing saved model: Saved model does not appear to be valid")
            raise err
    else:
        pytest.skip("Model file not found.")

@pytest.fixture(scope="module")
def train_dataset(data):
    train, _ = train_test_split(data, test_size=0.20, random_state=10, stratify=data['salary'])
    X_train, y_train, _, _ = process_data(train, categorical_features=CAT_FEATURES, label="salary", training=True)
    return X_train, y_train

def test_import_data(data):
    assert data.shape[0] > 0
    assert data.shape[1] > 0

def test_features(data):
    assert sorted(set(data.columns).intersection(CAT_FEATURES)) == sorted(CAT_FEATURES)

def test_is_model(model):
    # The fixture will handle the existence and validity of model.
    pass

def test_is_fitted_model(model, train_dataset):
    X_train, _ = train_dataset
    try:
        model.predict(X_train)
    except NotFittedError as err:
        logging.error(f"Model is not fit, error {err}")
        raise err

def test_inference(model, train_dataset):
    X_train, _ = train_dataset
    try:
        preds = inference(model, X_train)
    except Exception as err:
        logging.error("Inference cannot be performed on saved model and train data")
        raise err

def test_compute_model_metrics(model, train_dataset):
    X_train, y_train = train_dataset
    preds = inference(model, X_train)
    try:
        precision, recall, fbeta = compute_model_metrics(y_train, preds)
    except Exception as err:
        logging.error("Performance metrics cannot be calculated on train data")
        raise err

def test_compute_confusion_matrix(model, train_dataset):
    X_train, y_train = train_dataset
    preds = inference(model, X_train)
    try:
        cm = compute_confusion_matrix(y_train, preds)
    except Exception as err:
        logging.error("Confusion matrix cannot be calculated on train data")
        raise err
