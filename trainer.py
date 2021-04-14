import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff

# import numpy as np
# import argparse
# import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from datetime import datetime


# define constants
# N_FEATURES = None # the number of the features
N_CLIENTS = 5 # the total number of the clients
TEST_FRAC = 0.1 # The fraction of the complete dataset that will be taken for the test set
N_CLASSES = 2 # ATTACK / BENIGN

LEARNING_RATE = 0.06
BATCH_SIZE = 32
N_EPOCHS = 100 # the number of epochs (times the dataset will be repeated)
SHUFFLE_BUFFER = 100
N_ROUNDS = 100 # The number of the federated training rounds

train = list() # the list with the client train sets
test = list() # the list with the client test sets

REPORT_FILENAME = f"executions/{datetime.now().strftime('%d-%m-%Y_%H.%M.%S')}.txt" # the name of the report file



# features selected during feature selection
USEFUL_FEATURES = ["Bwd Packet Length Mean", "Avg Bwd Segment Size",  "Idle Min", "Flow IAT Max", "Fwd IAT Max",
                   "Bwd Packet Length Std", "Average Packet Size", "Idle Mean", "min_seg_size_forward",
                   "Packet Length Mean", "Idle Max", "Max Packet Length",  "Fwd Packets/s", "Min Packet Length",
                   "Fwd IAT Total", "Fwd IAT Std",  "Bwd Packet Length Max", "Packet Length Std", "Flow IAT Std",
                   "Fwd Packet Length Min", "Flow Packets/s"]


def selectFeatures(dataset):
    '''
    Select a subset of features from a given dataset
    :param dataset (pandas.DataFrame): The original dataset
    :return (pandas.DataFrame): The dataset containing only a subset of the original features
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


def defineModel(write_file=False):
    '''
    Defines a keras model and its architecture
    :param write_file (Boolean): If True then it will write the architecture of the model to the report file
    :return (keras Sequential Model): The keras model
    '''
    # define the model's architecture
    model = keras.models.Sequential()
    model.add(tf.keras.Input(shape=[train[0].element_spec[0].shape.dims[1]], name="Input"))
    model.add(tf.keras.layers.Dense(26, activation='relu', name="Hidden_1"))
    model.add(tf.keras.layers.Dense(21, activation='relu', name="Hidden_2"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid", name="Output"))


    # write the architecture of the network to the file
    if write_file:

        f = open(REPORT_FILENAME, "a+")
        f.write("\n\n----- NETWORK ARCHITECTURE -----\n")
        f.write(f"HIDDEN LAYERS: {str(len(model.layers)-1)}\n")
        for l in model.layers:
            neurons = l.units
            if l.name != "Output":
                f.write(f"\tSHAPE: {neurons}, ACTIVATION FUNCTION: RELU\n")

        f.close()

    return model


def createModel():
    '''
    Defines the network's architecture (layers, shapes, activation functions, losses and metrics and builds the model
    :return (tff.learning.Model): The built model
    '''
    # define the model's architecture
    model = defineModel()

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

    """ --- PREPROCESSING ---"""

    # load the client datasets
    for i in range(N_CLIENTS):
        tmp_dataset = loadDataset(f"client{i}.csv")
        tmp_dataset = selectFeatures(tmp_dataset)

        # seperate x from y
        x = tmp_dataset.iloc[:, :len(tmp_dataset.columns)-1]
        x_cols = x.columns
        y = tmp_dataset.iloc[:, -1]

        # split the dataset into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_FRAC, random_state=42)

        # scale the dataset
        x_train_scaled, x_test_scaled = scaleX(x_train, x_test)

        # transform the train and test sets to TF datasets after you preprocess them first
        d = tf.data.Dataset.from_tensor_slices((x_train_scaled.values, y_train.values))
        train.append(preprocess(d))
        d = tf.data.Dataset.from_tensor_slices((x_test_scaled.values, y_test.values))
        test.append(preprocess(d))


    # write the features to a file
    f = open(REPORT_FILENAME, "w+")
    f.write("---- FEATURES ----\n")
    f.write(str(USEFUL_FEATURES))
    f.close()

    # write the model's architecture to a file
    defineModel(write_file=True)


    """ --- TRAINING ---"""

    # create the Federated Averaging process
    trainer = tff.learning.build_federated_averaging_process(createModel, client_optimizer_fn=
                                    lambda: tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE))


    print("Starting Training ...")

    f = open(REPORT_FILENAME, "a+")
    f.write("\n\n----- TRAINING -----\n")
    f.close()

    # perform N_ROUNDS of training
    state = trainer.initialize()
    for i in range(N_ROUNDS):
        state, metrics = trainer.next(state, train)
        report_string = f"round {i} -> Loss =  {metrics['train']['loss']}, Accuracy = {metrics['train']['accuracy']}"
        print(report_string)

        # write report to file
        f = open(REPORT_FILENAME, "a+")
        f.write(report_string + "\n")
        f.close()

    print("[+] Training Finished")

    """ --- EVALUATION ---"""
    f = open(REPORT_FILENAME, "a+")
    f.write("\n\n----- TEST -----\n")

    # evaluate the model
    evaluation = tff.learning.build_federated_evaluation(createModel)
    eval_metrics = evaluation(state.model, test)
    report_string = f"\nLoss = {eval_metrics['loss']}, Accuracy = {eval_metrics['accuracy']}"

    # print results
    print(report_string)
    f.write(report_string + "\n")
    f.close()

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