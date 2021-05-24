# load required packages

import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
import os
import matplotlib.pyplot as plt
import argparse
# import numpy as np


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from datetime import datetime
from checkpoint_manager import FileCheckpointManager


# define constants
# N_FEATURES = None # the number of the features
N_CLIENTS = 5    # the total number of the clients
TEST_FRAC = 0.1    # The fraction of the complete dataset that will be taken for the test set
N_CLASSES = 2    # ATTACK / BENIGN

LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 100  # the number of epochs (times the dataset will be repeated)
SHUFFLE_BUFFER = 130000
N_ROUNDS = 10    # The number of the federated training rounds

train = list()  # the list with the client train sets
test = list()   # the list with the client test sets

# define constants for reporting
DATE = datetime.now().strftime('%d-%m-%Y_%H.%M.%S')
EXECUTIONS_DIR = "executions"
REPORT_EXTENSION = "txt"
REPORT_FILENAME = f"{EXECUTIONS_DIR}/{DATE}/{DATE}.{REPORT_EXTENSION}"  # the name of the report file



# features selected during feature selection
# these features were selected using Feature Importance (ExtraTreesClassifier and further discarding the
# features with low variance
USEFUL_FEATURES = ["Bwd Packet Length Mean", "Avg Bwd Segment Size", "Flow IAT Max", "Fwd IAT Max",
                   "Bwd Packet Length Std", "Idle Mean", "min_seg_size_forward",
                   "Packet Length Mean", "Max Packet Length",  "Fwd Packets/s", "Min Packet Length",
                   "Fwd IAT Total", "Fwd IAT Std",  "Bwd Packet Length Max", "Packet Length Std", "Flow IAT Std",
                   "Fwd Packet Length Min", "Flow Packets/s"]


def getCMDArgs():
    '''
    Parses the arguments given in the terminal
    :return: The arguments' values
    '''

    # define parser
    parser = argparse.ArgumentParser(description="Trains and evaluates a classifier for classifying whether a network "
                                                 "flor is intrusive or normal",
                                     usage="python trainer.py [OPTION] ... [FILENAME]")

    # add arguments
    parser.add_argument("-m", "--model", help="Load and evaluate a pre-trained model. Add the path to the pre-trained "
                                   "model's directory")

    args = parser.parse_args()
    return args.model


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


def standardizeX(x_train, x_test=None):
    '''
    Standardizes the features. If passed with a test set then, the test set is standardized using the
    training set.
    :param x_train (pandas.DataFrame): The training x (features)
    :param x_test (pandas.DataFrame): (optional) The test x (features)
    :return (pandas.DataFrame(s)): The standardized dataset(s)
    '''
    standardizer = StandardScaler().fit(x_train)

    x_cols = x_train.columns

    x_train_std = pd.DataFrame(standardizer.transform(x_train), columns=x_cols)

    if x_test is not None:
        x_test_std = pd.DataFrame(standardizer.transform(x_test), columns=x_cols)
        return x_train_std, x_test_std

    return x_train_std


def defineModel(input_shape, write_file=False):
    '''
    Defines a keras model and its architecture
    :param write_file (Boolean): If True then it will write the architecture of the model to the report file
    :return (keras Sequential Model): The keras model
    '''
    hidden_activation_fn = "selu"
    # define the model's architecture
    model = keras.models.Sequential()
    model.add(tf.keras.Input(shape=[input_shape], name="Input"))
    model.add(tf.keras.layers.Dense(200, activation=hidden_activation_fn, kernel_initializer="lecun_normal",
                                    name="Hidden_1"))
    model.add(tf.keras.layers.Dense(150, activation=hidden_activation_fn, kernel_initializer="lecun_normal",
                                    name="Hidden_2"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid", name="Output"))


    # write the architecture of the network to the file
    if write_file:

        f = open(REPORT_FILENAME, "a+")
        f.write("\n\n----- NETWORK ARCHITECTURE -----\n")
        f.write(f"HIDDEN LAYERS: {str(len(model.layers)-1)}\n")
        for l in model.layers:
            neurons = l.units
            hidden_activation_fn = l.output.name
            if l.name != "Output":
                f.write(f"\tSHAPE: {neurons}, ACTIVATION FUNCTION: {hidden_activation_fn}\n")

        f.close()

    return model


def createModel():
    '''
    Defines the network's architecture (layers, shapes, activation functions, losses and metrics and builds the model
    :return (tff.learning.Model): The built model
    '''
    # define the model's architecture
    model = defineModel(train[0].element_spec[0].shape.dims[1])

    # return model
    return tff.learning.from_keras_model(model,
                                         input_spec=train[0].element_spec,
                                         loss=tf.losses.BinaryCrossentropy(),
                                         metrics=[tf.keras.metrics.Accuracy()])


def preprocess(dataset, n_reps=N_EPOCHS, shuffle_buffer=SHUFFLE_BUFFER, batch_size=BATCH_SIZE):
    '''
    Preprocesses the data. Repeats, shuffles and creates batches of a given dataset
    :param dataset (pandas.DataFrame): The dataset which will be preprocessed
    :return (pandas.DataFrame): The processed dataset
    '''
    return dataset.repeat(n_reps).shuffle(shuffle_buffer).batch(batch_size).prefetch(1)


def plotHistory(history):
    """
    Plots the training curves based on the loss and accuracy of the model
    :param history (pd.DataFrame): The frame containing the training metrics
    """
    pd.DataFrame(history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.savefig(f"{EXECUTIONS_DIR}/{DATE}/training_curves.png")

    plt.show()


def main():

    # parse the cmd arguments
    args = getCMDArgs()

    """ --- PREPROCESSING ---"""
    # load the client datasets
    for i in range(N_CLIENTS):
        tmp_dataset = loadDataset(f"Corrected_Datasets/client_{i}.csv")
        tmp_dataset = selectFeatures(tmp_dataset)

        # seperate label (y) from features (x)
        x = tmp_dataset.drop("Label", axis="columns")
        y = tmp_dataset["Label"].copy()

        # split the dataset into train and test sets
        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_FRAC, random_state=42)

        # split train and test set (STRATIFIED)
        split = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=42)
        for tr_index, test_index in split.split(tmp_dataset, tmp_dataset["Label"]):
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

        # standardize the dataset
        # x_train_scaled, x_test_scaled = standardizeX(x_train, x_test)

        # scale the dataset
        # x_train_scaled, x_test_scaled = scaleX(x_train, x_test)
        pd.set_option('display.max_columns', 30)
        # print(x_train_scaled.describe())
        # standardize the dataset
        x_train_scaled, x_test_scaled = standardizeX(x_train, x_test)

        print(x_train_scaled.describe())

        # transform the train and test sets to TF datasets after you preprocess them first
        d = tf.data.Dataset.from_tensor_slices((x_train_scaled.values, y_train.values))
        train.append(preprocess(d))
        # test set does not need to be repeated only one instance is enough
        d = tf.data.Dataset.from_tensor_slices((x_test_scaled.values, y_test.values))
        test.append(preprocess(d, n_reps=1))


    # create and initialize the Federated Averaging process
    trainer = tff.learning.build_federated_averaging_process(createModel, client_optimizer_fn=
                                    lambda: tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))

    state = trainer.initialize()


    """ --- TRAINING ---"""

    if args is not None:  # evaluate only a pre-trained model
        # load the pre-trained model
        fcm = FileCheckpointManager(args, prefix="saved_model")
        state, _ = fcm.load_latest_checkpoint(state)

    else:   # train and evaluate a model

        # create report directory
        try:
            os.mkdir(f"{EXECUTIONS_DIR}/{DATE}")
        except OSError as oserr:
            print("\n[-] Creation of report directory failed. Exiting...")
            return 1
        else:
            print("\n[+] Report directory was created successfully.")

        # write the features to a file
        f = open(REPORT_FILENAME, "w+")
        f.write("---- FEATURES ----\n")
        f.write(str(USEFUL_FEATURES))
        f.close()

        # write the model's architecture to a file
        defineModel(train[0].element_spec[0].shape.dims[1], write_file=True)

        # write the optimizer to the report file
        f = open(REPORT_FILENAME, "a+")
        f.write("\n\n---- OPTIMIZER ----\n")
        f.write(f"\tOptimizer: Adam\n\tLearning Rate: {LEARNING_RATE}")
        f.close()

        print("Starting Training ...")

        f = open(REPORT_FILENAME, "a+")
        f.write("\n\n----- TRAINING -----\n")
        f.close()

        history = pd.DataFrame(columns=["Loss", "Accuracy"])

        # perform N_ROUNDS of training
        for i in range(N_ROUNDS):
            state, metrics = trainer.next(state, train)
            report_string = f"round {i} -> Loss =  {metrics['train']['loss']}, Accuracy = {metrics['train']['accuracy']}"
            print(report_string)
            history = history.append({"Loss": metrics['train']['loss'], "Accuracy": metrics['train']['accuracy']}
                                     , ignore_index=True)

            # write report to file
            f = open(REPORT_FILENAME, "a+")
            f.write(report_string + "\n")
            f.close()

        plotHistory(history)
        print("[+] Training Finished")

        # save model as a checkpoint
        fcm = FileCheckpointManager(f'{EXECUTIONS_DIR}/{DATE}/', prefix=f"saved_model_{DATE}_")
        fcm.save_checkpoint(state, round_num=N_ROUNDS)

    """ --- EVALUATION ---s"""

    # evaluate the model
    evaluation = tff.learning.build_federated_evaluation(createModel)
    eval_metrics = evaluation(state.model, test)
    report_string = f"\nLoss = {eval_metrics['loss']}, Accuracy = {eval_metrics['accuracy']}"

    # print results
    print(report_string)

    if args is None:    # write to file only if training a model
        f = open(REPORT_FILENAME, "a+")
        f.write("\n\n----- TEST -----\n")
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
