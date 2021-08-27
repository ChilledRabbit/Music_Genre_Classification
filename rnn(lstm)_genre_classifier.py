import json
import numpy as np
import sklearn.model_selection
import tensorflow.keras as keras
import tensorflow as tf

DATASET_PATH = "data.json"
print(tf.__version__)


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

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):

    # Create model

    model = keras.Sequential()

    # LSTM Layers

    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))

    # Dense Layer

    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # Output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model

def predict(model, X, y):

    X = X[np.newaxis, ...]

    prediction = model.predict(X)  # Prediction will be array with values for the probability of each genre.

    # Extract index with max value

    predicted_index = np.argmax(prediction, axis=1)
    print("Expected index: {}, Predicted index: {}".format(y, predicted_index))


if __name__ == "__main__":
    # Create train, validation and test sets

    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_dataset(0.25, 0.2)

    # Build the RNN-LSTM network

    input_shape = (X_train.shape[1], X_train.shape[2])  # 130, 13
    model = build_model(input_shape)

    # Compile the network

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy']
                  )

    # Train RNN

    model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)

    # Evaluate the RNN on the test set

    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))

    # Export Model

    model.save('My model')

    # Make prediction on a sample

    X = X_test[500]
    y = y_test[500]
    predict(model, X, y)
