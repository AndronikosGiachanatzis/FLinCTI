import pandas as pd
import numpy as np
import argparse
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow_federated as tff

# define constants
N_FEATURES = None
TEST_FRAC = 0.1
N_CLASSES = 2 # ATTACK / BENIGN
LEARNING_RATE = 0.3
BATCH_SIZE = 32
TRAINING_STEPS = 1000
N_EPOCHS = 100
SHUFFLE_BUFFER = 100
N_ROUNDS = 500
train = list()
test = list()

# features selected during feature selection
USEFUL_FEATURES = ["ACK Flag Count", "Bwd Packet Length Mean", "Avg Bwd Segment Size",
                   "Idle Min", "Flow IAT Max", "Fwd IAT Max", "Bwd Packet Length Std", "Average Packet Size",
                   "Idle Mean", "min_seg_size_forward", "Packet Length Mean", "Idle Max", "Max Packet Length",
                   "Fwd Packets/s", "Min Packet Length", "Fwd IAT Total", "FIN Flag Count", "Fwd IAT Std",
                   "Bwd Packet Length Max", "URG Flag Count", "Packet Length Std", "Flow IAT Std",
                   "Fwd Packet Length Min", "Flow Packets/s"]


def selectFeatures(dataset):
    '''
    Select a subset of features from a given dataset
    :param dataset (pandas.DataFrame): The original dataset
    :return (pandas.DataFrame): The dataset containing only a subset of the original featuress\
    '''

    return dataset[USEFUL_FEATURES + ["Label"]]


def loadDataset(name):
    '''
    Load a csv dataset from memory
    :return (pandas.DataFrame): The loaded dataset
    '''
    # load the datast
    print("Loading dataset...")
    dataset = pd.read_csv(name)
    print("[+] Dataset loaded successfully")

    return dataset


def scaleX(x_train, x_test=None):
    '''
    Scales the features to fit in the range [0,1]. If passed with a test set then, the test set is scaled using the
    training set.
    :param x_train (pandas.DataFrame): The training x (features)
    :param x_test (pandas.DataFrame): (optional) The test x (features)
    :return (pandas.DataFrame(s)): The scaled dataset(s)
    '''
    # define the scaler
    scaler = MinMaxScaler().fit(x_train)
    # get the column names from the dataset
    x_cols = x_train.columns

    # scale the training set
    x_train_scaled = pd.DataFrame(scaler.transform(x_train), columns=x_cols)

    # scale the dataset if passed as an argument
    if x_test is not None:
        x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_cols)
        return x_train_scaled, x_test_scaled

    return x_train_scaled


def createModel():
    '''
    Defines the network's architecture (layers, shapes, activation functions, losses and metrics and builds the model
    :return (tff.learning.Model): The built model
    '''
    # the model architecture, number and shapes of layers, activation functions
    model = keras.models.Sequential()
    model.add(tf.keras.Input(shape=[24], name="Input"))
    model.add(tf.keras.layers.Dense(20, activation='relu', name="Hidden_1"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid", name="Output"))
    # return model
    return tff.learning.from_keras_model(model,
                                         input_spec=train[0].element_spec,
                                         loss=tf.losses.BinaryCrossentropy(),
                                         metrics=[tf.keras.metrics.Accuracy()])


def preprocess(dataset):
    '''
    Preprocesses the data. Repeats, shuffles and creates batches of a given dataset
    :param dataset (pandas.DataFrame): The dataset which will be preprocessed
    :return (pandas.DataFrame): The processed dataset
    '''
    return dataset.repeat(N_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)


def main():

    # load the client datasets
    for i in range(5):
        tmp_dataset = loadDataset(f"client{i}.csv")
        tmp_dataset = selectFeatures(tmp_dataset)
        N_FEATURES = tmp_dataset.columns.size - 1

        # seperate x from y
        x = tmp_dataset.iloc[:, :len(tmp_dataset.columns)-1]
        x_cols = x.columns
        y = tmp_dataset.iloc[:, -1]

        # split the dataset into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_FRAC, random_state=42)

        # scale the dataset
        x_train_scaled, x_test_scaled = scaleX(x_train, x_test)

        # transform the train and test sets to TF datasets
        d = tf.data.Dataset.from_tensor_slices((x_train_scaled.values, y_train.values))
        train.append(preprocess(d))
        d = tf.data.Dataset.from_tensor_slices((x_test_scaled.values, y_test.values))
        test.append(d)

    # create the Federated Averaging process
    trainer = tff.learning.build_federated_averaging_process(createModel, client_optimizer_fn=
                                    lambda: tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE))

    # perform N_ROUNDS of training
    state = trainer.initialize()
    for i in range(N_ROUNDS):
        state, metrics = trainer.next(state, train)
        print(f"round {i} -> Loss =  {metrics['train']['loss']}, Accuracy = {metrics['train']['accuracy']}")

    # evaluate the model
    evalulation = tff.learning.build_federated_evaluation(createModel)
    eval_metrics = evalulation(state.model, test)
    print(eval_metrics)

    # print(dataset.info())
    #
    # dataset = selectFeatures(dataset)
    # # split into x and y
    # x = dataset.iloc[:, :len(dataset.columns)-1]
    # x_cols = x.columns
    # y = dataset.iloc[:, -1]
    #
    # # split into train and test set
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    #
    # # scale the dataset
    # x_train_scaled, x_test_scaled = scaleX(x_train, x_test)
    # train_scaled = pd.concat([x_train_scaled, y_train], axis=1)
    # N_TRAIN_SAMPLES = len(train_scaled)
    # print()



    # from sklearn.linear_model import SGDClassifier
    #
    # clf = SGDClassifier(random_state=42)
    #
    # clf.fit(x_train_scaled, y_train)
    # preds = clf.predict(x_test_scaled)
    # correct = sum(preds == y_test)
    # from sklearn.metrics import confusion_matrix
    # from sklearn.model_selection import cross_val_predict
    # print(confusion_matrix(y_test, clf.predict(x_test_scaled)))

    # ------------------- METHOD 2 ----------------------
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.metrics import classification_report, confusion_matrix
    #
    # model = LogisticRegression(solver="liblinear", random_state=42)
    # model.fit(x_train_scaled, y_train)
    # preds = model.predict(x_test_scaled)
    # # print(confusion_matrix(y_test, preds))
    # print(model.score(x_train_scaled, y_train))

    # --------------- KERAS ---------------

    # tf_train = tf.data.Dataset.from_tensor_slices((x_train_scaled.values, y_train.values))
    #
    # model = createModel(N_FEATURES)
    #
    # model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(), metrics=["accuracy"])
    #
    # history = model.fit(tf_train, epochs=N_EPOCHS)
    #
    #
    # pd.DataFrame(history.history).plot(figsize=(8,5))
    # plt.grid(True)
    # plt.gca().set_ylim(0, 1)
    # plt.show()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("[!] Keyboard Interrupt detected. Exiting...")
        exit(0)