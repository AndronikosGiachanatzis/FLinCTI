import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# define constants
N_IMPORTANT_FEATURES = 25
VARIANCE_THRESHOLD = 0

# features selected during feature selection
USEFUL_FEATURES = ["ACK Flag Count", "Bwd Packet Length Mean", "Avg Bwd Segment Size",
                   "Idle Min", "Flow IAT Max", "Fwd IAT Max", "Bwd Packet Length Std", "Average Packet Size",
                   "Idle Mean", "min_seg_size_forward", "Packet Length Mean", "Idle Max", "Max Packet Length",
                   "Fwd Packets/s", "Min Packet Length", "Fwd IAT Total", "FIN Flag Count", "Fwd IAT Std",
                   "Bwd Packet Length Max", "URG Flag Count", "Packet Length Std", "Flow IAT Std",
                   "Fwd Packet Length Min", "Flow Packets/s"]


def loadDataset(name):
    '''
    Select a subset of features from a given dataset
    :param dataset (pandas.DataFrame): The original dataset
    :return (pandas.DataFrame): The dataset containing only a subset of the original features
    '''

    # load the datast
    print("Loading dataset...")
    dataset = pd.read_csv(name)
    print("[+] Dataset loaded successfully")

    return dataset

def selectFeatures(dataset):
    '''
    Select a subset of features from a given dataset
    :param dataset (pandas.DataFrame): The original dataset
    :return (pandas.DataFrame): The dataset containing only a subset of the original featuress\
    '''

    return dataset[USEFUL_FEATURES + ["Label"]]


def makeBinary(dataset):
    '''
    Transforms the dataset's labels to binary BENIGN/ATTACK
    :param dataset (pandas.DataFrame): the Dataset to be transformed
    :return (pandas.DataFrame): The tranformed dataset
    '''
    return dataset['Label'].replace(['DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris', 'DoS GoldenEye', 'Heartbleed'],
                                    'ATTACK')


def writeFrameToCsv(dataset, filename):
    '''
    Writes the dataset to a csv
    :param dataset (pandas.DataFrame): The dataset to be written
    :param filename (String): The name of the output file
    '''
    dataset.to_csv(path_or_buf=filename, index=False)


def cleanDataset(dataset):
    '''
    Delete all rows containing either infinite (inf, -inf) or missing values (NaN)
    :param dataset (pandas.DataFrame): The dataset to be edited
    :return: The dataset minus the rows that contained inf, -inf or missiing values
    '''
    # remove infinite values and drop missing values
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()
    dataset = dataset.reset_index(drop=True)
    return dataset


def transformLabelToNumber(dataset):
    '''
    Encodes the labels of the dataset to numerical
    :param dataset (pandas.DataFrame): The dataset to be edited
    :return (pandas.DataFrame): The dataset with numerical labels
    '''
    encoder = OrdinalEncoder()
    dataset = encoder.fit_transform(dataset.values.reshape(-1,1))
    return dataset


def univariateSelection(x,y):
    '''
    Performs Univariate Feature Selection. Selects the best features in the dataset
    :param x (pandas.DataFrame): The independent variables - feautres
    :param y (pandas.DataFrame): The dependent variables - labels
    '''
    # f_class: ANOVA F-value between label/feature for classification tasks
    bestFeatures = SelectKBest(score_func=f_classif, k=N_IMPORTANT_FEATURES)
    fit = bestFeatures.fit(x, y)

    print(getFeatureSelectionResults(fit, x.columns))



def getFeatureSelectionResults(fit, columns):
    '''

    :param fit:
    :param columns:
    :return:
    '''
    scores = pd.DataFrame(fit.scores_)
    columns = pd.DataFrame(columns)

    featureScores = pd.concat([columns, scores], axis=1)
    featureScores.columns = ['Specs', 'Score']
    f = open("bestFeatures-Univariate_Selection.txt", "w")

    f.write(featureScores.nlargest(N_IMPORTANT_FEATURES, 'Score')['Specs'].to_string())
    f.close()
    return featureScores.nlargest(N_IMPORTANT_FEATURES, 'Score')


def featureImportance(x,y):
    '''
    Performs Feature Importance Analysis for Feature Selection using the ExtraTreesClassifier
    :param x (pandas.DataFrame): The independent variables - feautres
    :param y (pandas.DataFrame): The dependent variables - labels
    :return: The scores of the analysis
    '''
    model = ExtraTreesClassifier()
    model.fit(x, y)

    scores = pd.DataFrame(model.feature_importances_)
    columns = pd.DataFrame(x.columns)
    featureScores = pd.concat([columns, scores], axis=1)
    featureScores.columns = ['Specs', 'Score']

    f = open("bestFeatures-Feature_Importance.txt", "w")

    f.write(featureScores.nlargest(N_IMPORTANT_FEATURES, 'Score')['Specs'].to_string())
    f.close()
    return model.feature_importances_


def heatMap(dataset):
    '''
    Generates a correlation heatmap for the dataset's variables
    :param dataset: The variables of which the correlations will be found
    '''
    corrmat = dataset.corr()
    top_corr_feats = corrmat.index

    # draw plot
    plt.figure(figsize=(30, 30))
    heatmap = sns.heatmap(dataset[top_corr_feats].corr(), annot=True, cmap="RdYlGn")

    # save plot
    plt.savefig('heatmap.png')


def removeLowVariances(x):
    '''
    Removes all variables from the dataset that have variance below a given threshold
    :param x (pandas.DataFrame): The variables whose variance will be analyzed
    :return (pandas.DataFrame): The variables whose variance surpasses the threshold
    '''
    selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    selector.fit(x)
    new_x = x[x.columns[selector.get_support(indices=True)]]
    return new_x


def kendalls(dataset):
    '''
    Creates a kendall's coefficient heatmap for the variables of the dataset
    :param dataset (pandas.DataFrame): The variables of which the kendall's coefficients will be found
    '''
    import seaborn as sns

    corrmat = dataset.corr()
    top_corr_feats = corrmat.index
    plt.figure(figsize=(50, 50))

    heatmap = sns.heatmap(dataset[top_corr_feats].corr(method="kendall"), annot=True, cmap="RdYlGn")
    plt.savefig('heatmap-kendall.png')


def rfecv(x, y):
    clf = DecisionTreeClassifier()
    trans = RFECV(clf)
    x_trans = trans.fit_transform(x, y)
    return trans


def SequentialFeatureSelection(x, y):
    '''
    Perform Forward Sequential Feature Selection to find the best subset of features
    :param x (pandas.DataFrame): The independent variables - features of the dataset
    :param y (pandas.DataFrame): The dependent variables - labels of the dataset
    :return : The scores of the analysis
    '''
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import KFold
    # define the classifier that will be used
    classifier = KNeighborsClassifier(n_neighbors=10)

    cv = KFold(n_splits=10, shuffle=False)

    sfs = SFS(classifier, k_features=20, forward=True, scoring="accuracy", cv=cv)

    sfs = sfs.fit(x, y)
    return sfs


def main():
    dataset = loadDataset("binaryDatasetCleanEncodedNoZeros.csv")

    x = dataset.iloc[:, :len(dataset.columns)-1]
    x_cols = x.columns
    y = dataset.iloc[:, -1]

    sub = SequentialFeatureSelection(x, y)
    print()


    # dataset = selectFeatures(dataset)
    # heatMap(dataset)

    # print(abs(dataset.corr()["Bwd Packet Length Mean"]))



    # NUM_CLIENTS = 6

    # datasets = list()
    # for i in range(5):
    #     tmp_dataset = dataset.sample(frac=0.2, random_state=42)
    #     tmp_dataset = tmp_dataset.reset_index(drop=True)
    #     print(tmp_dataset)
    #     writeFrameToCsv(tmp_dataset, f"client{i}.csv")



    # x = dataset.iloc[:, :len(dataset.columns)-1]
    # x_cols = x.columns
    # y = dataset.iloc[:, -1]
    #
    # # split into train and test set
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    # print(x_train.describe())


    # scale sets
    # scaler = MinMaxScaler().fit(x_train)
    #
    # x_train_scaled = pd.DataFrame(scaler.transform(x_train), columns=x_cols)
    # x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_cols)
    #
    # dataset_scaled = pd.concat([x_train_scaled, y_train], axis=1)




    # univariateSelection(x_train_scaled, y_train)
    # featureImportance(x_train_scaled, y_train)
    # heatMap(dataset_scaled)
    # print(dataset_scaled.describe())
    # trans = rfecv(x,y)
    # columns_retained = dataset.iloc[:, 1:].columns[trans.get_support()].values




if __name__ == "__main__":
    main()