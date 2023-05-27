import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def separate_features_from_label(X, label=None):
    """
    If a label is provided, separate the features and labels in the data.
    """
    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    return X, y


def split_categorical_and_continuous(X, categorical_features=[]):
    """
    Split the data into categorical and continuous features.
    """
    X_categorical = X[categorical_features].values
    X_continuous = X.drop(columns=categorical_features, axis=1)

    return X_continuous, X_categorical


def handle_categorical_features(X_categorical, training=True, encoder=None):
    """
    Use one hot encoding for the categorical features.
    """
    if training:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        X_categorical = encoder.fit_transform(X_categorical)
    else:
        X_categorical = encoder.transform(X_categorical)

    return X_categorical, encoder


def handle_labels(y, training=True, lb=None):
    """
    Use label binarizer for the labels.
    """
    if y.size != 0:
        if training:
            lb = LabelBinarizer()
            y = lb.fit_transform(y.values).ravel()
        else:
            y = lb.transform(y.values).ravel()

    return y, lb


def process_data(
        X,
        categorical_features=[],
        label=None,
        training=True,
        encoder=None,
        lb=None
):
    """
    Process the data used in the machine learning pipeline.
    """

    X, y = separate_features_from_label(X, label)
    X_continuous, X_categorical = split_categorical_and_continuous(X, categorical_features)
    X_categorical, encoder = handle_categorical_features(X_categorical, training, encoder)
    y, lb = handle_labels(y, training, lb)

    X = np.concatenate([X_continuous, X_categorical], axis=1)

    return X, y, encoder, lb
