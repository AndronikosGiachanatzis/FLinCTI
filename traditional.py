import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow import keras
from trainer import scaleX
from sklearn.preprocessing import MinMaxScaler
from trainer import loadDataset
from trainer import USEFUL_FEATURES
from trainer import selectFeatures
from trainer import TEST_FRAC
from trainer import LEARNING_RATE
from trainer import BATCH_SIZE
from trainer import defineModel


# def defineModel(n_features):
#
#     model = keras.models.Sequential()
#     model.add(keras.layers.InputLayer(input_shape=[n_features]))
#     model.add(keras.layers.Dense(150, kernel_initializer="he_normal", activation='relu', name="Hidden_1"))
#     model.add(keras.layers.Dense(150, kernel_initializer="he_normal", activation='relu', name="Hidden_2"))
#     model.add(keras.layers.Dense(1, kernel_initializer="he_normal", activation="sigmoid", name="Output"))
#
#     return model


def plotHistory(history):
    """
    Plots the training curves based on the loss and accuracy of the model
    :param history (pd.DataFrame): The frame containing the training metrics
    """
    pd.DataFrame(history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)

    plt.show()


def main():
    # load dataset
    dataset = loadDataset("binaryDatasetCleanEncodedCorrectedNoZeros.csv")
    dataset = selectFeatures(dataset)

    # seperate label (y) from features (x)
    x = dataset.drop("Label", axis="columns")
    y = dataset["Label"].copy()

    # split train and test set (STRATIFIED)
    split = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=42)
    for tr_index, test_index in split.split(dataset, dataset["Label"]):
        # train sets
        x_train = x.loc[tr_index]
        y_train = y.loc[tr_index]

        # test sets
        x_test = x.loc[test_index]
        y_test = y.loc[test_index]

    # reset indices
    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # scale the dataset
    x_train_scaled, x_test_scaled = scaleX(x_train, x_test)


    # TRAINING
    model = defineModel(len(USEFUL_FEATURES))

    model.compile(loss=tf.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  metrics=["accuracy"])

    early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)

    history = model.fit(x_train_scaled, y_train, epochs=5,
                        validation_data=(x_test_scaled, y_test), batch_size=BATCH_SIZE,  callbacks=[early_stopping_cb])


    plotHistory(history.history)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Keyboard Interrupt detected. Exiting...")
        exit(0)