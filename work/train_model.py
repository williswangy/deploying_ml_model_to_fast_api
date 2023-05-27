import os
import pickle
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from work.ml.data import process_data

from work.ml.model import train_model, compute_model_metrics, inference, compute_slices, compute_confusion_matrix

logging.basicConfig(filename='journal.log', level=logging.INFO, filemode='a', format='%(name)s - %(levelname)s - %(message)s')

def remove_file_if_exists(filename):
    if os.path.exists(filename):
        os.remove(filename)

def load_data(path):
    return pd.read_csv(path)

def split_data(data, target='salary', test_size=0.20, random_state=10):
    return train_test_split(data, test_size=test_size, random_state=random_state, stratify=data[target])

def model_exists(path, filename):
    return os.path.isfile(os.path.join(path, filename))

def load_model(savepath, filenames):
    return [pickle.load(open(os.path.join(savepath, filename), 'rb')) for filename in filenames]

def train_and_save_model(X, y, savepath, filenames):
    model = train_model(X, y)
    for item, filename in zip([model, encoder, lb], filenames):
        pickle.dump(item, open(os.path.join(savepath, filename), 'wb'))
    logging.info(f"Model saved to disk: {savepath}")
    return model

def evaluate_and_log_model(model, X_test, y_test, lb):
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    logging.info(f"Classification target labels: {list(lb.classes_)}")
    logging.info(f"precision:{precision:.3f}, recall:{recall:.3f}, fbeta:{fbeta:.3f}")

    cm = compute_confusion_matrix(y_test, preds, labels=list(lb.classes_))
    logging.info(f"Confusion matrix:\n{cm}")
    return preds  # returning the preds

def compute_slices_and_save_to_file(data, features, y_test, preds, savepath):
    remove_file_if_exists(savepath)
    for feature in features:
        performance_df = compute_slices(data, feature, y_test, preds)
        performance_df.to_csv(savepath,  mode='a', index=False)
        logging.info(f"Performance on slice {feature}")
        logging.info(performance_df)

# Load the data
data = load_data("../data/census.csv")

# Split the data into train and test sets
train, test = split_data(data)

# Define the categorical features
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

# Process the training data
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Process the test data
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# Define the path to save the model
savepath = '../model'
filenames = ['trained_model.pkl', 'encoder.pkl', 'labelizer.pkl']

# Check if a trained model exists and either load or train a new model
if model_exists(savepath, filenames[0]):
    model, encoder, lb = load_model(savepath, filenames)
else:
    model = train_and_save_model(X_train, y_train, savepath, filenames)

# Evaluate and log the model
preds =evaluate_and_log_model(model, X_test, y_test, lb)

# Compute slices and save results to a file
compute_slices_and_save_to_file(test, cat_features, y_test, preds, "./slice_output.txt")