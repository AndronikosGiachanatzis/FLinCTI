import pandas as pd
import numpy as np
import argparse


def parseArguments():
    '''
    Parse the arguments given in the command line
    :return:
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", help="The id (number) of the client", type=int)
    args = parser.parse_args()

    client_id = args.n

    if client_id > 4 or client_id < 0:
        print ("[!] Invalid client id")
        return None

    return client_id

def selectFeatures(dataset):
    '''
    Select only the useful features for a DOS Slowloris attck detection according to the paper
    :return:
    '''
    useful_features_names = ["Flow Duration", "Flow IAT Min", "Bwd IAT Mean", "Flow IAT Mean"]

    return dataset[useful_features_names + ["Label"]]


def loadDataset(client_id):
    '''
    Load the dataset
    :return: A subset of the dataset comprised only by a subset of the original features
    '''
    # load the datast
    print("Loading dataset...")
    dataset = pd.read_csv(f"dataset_slowloris_{client_id}.csv")
    print("[+] Dataset loaded successfully")

    return dataset


def getDoSSamples(dataset):
    '''
    Get a random samples from the slowloris and benig samples
    :param dataset: The complete dataset
    :return: A smaller dataset containing only a subset of the DoS Slowloris and Benign samples
    '''

    dos_slowloris_label = "DoS slowloris"
    benign_label = "BENIGN"
    # select all the samples that are classified as a DoS slowloris attack
    dataset_slowloris = dataset.loc[dataset['Label'] == dos_slowloris_label]
    # select all the samples that are classified as benign
    dataset_benign = dataset.loc[dataset['Label'] == benign_label]

    # get a random sample from the datasets
    length = len(dataset_slowloris)
    sample_size = int(length*0.2)

    # sample
    slowloris = dataset_slowloris.sample(sample_size)
    print(slowloris)

    length = len(dataset_benign)
    sample_size = int(length*0.0030)

    # sample
    benign = dataset_benign.sample(sample_size)
    print(benign)

    frames = [slowloris, benign]
    return pd.concat(frames)




def main():

    client_id = parseArguments()
    if client_id is None:
        print("[-] Exiting...")
        return

    # load the dataset
    dataset = loadDataset(client_id)
    print(dataset)




if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("[!] Keyboard Interrupt detected. Exiting...")