import tensorflow
import numpy as np
import json
import sklearn.model_selection

DATASET_PATH = "data.json"

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # Convert lists into numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return X, y


def prepare_dataset(test_size, validation_size):

    # Load data

    X, y = load_data(DATASET_PATH)

    # Create train/test split

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=test_size)

    # Create the train validation split

    X_train, X_validation, y_train, y_validation = sklearn.model_selection.train_test_split(X_train, y_train, test_size=validation_size)

    # Make X_train and others a 3d array

    X_train = X_train[..., np.newaxis]  # 4d array -> (num_samples, 130, 13, 1) {X_train is actually 4d}
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def predict(model, X, y):

    X = X[np.newaxis, ...]

    prediction = model.predict(X)  # Prediction will be array with values for the probability of each genre.

    # Extract index with max value

    predicted_index = np.argmax(prediction, axis=1)
    print("Expected index: {}, Predicted index: {}".format(y, predicted_index))

# Prepare dataset

X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_dataset(0.25, 0.2)

# Predict using model generated in cnn_genre_classifier

model = tensorflow.keras.models.load_model('My RNN Model')  # Use 'My CNN Model' for prediction using CNN Model

X = X_test[100]
y = y_test[100]

predict(model, X, y)
