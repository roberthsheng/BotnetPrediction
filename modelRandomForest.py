import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import os
import pickle
import datetime
import itertools

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from scipy.sparse import csr_matrix

from utils import *
from utilsModel import *

PARENT_PATH = 'data'
TRAIN_FILE_NAME = f'{PARENT_PATH}/final_train.pkl'
TEST_FILE_NAME = f'{PARENT_PATH}/final_test.pkl'

saved_dict = {}
PREPROCESSED_PATH = 'data'

FIGURE_PATH = f'images'

def loadDataAndPrepFiles():
    
    x_train, y_train, y_train_label = pickle.load(open(TRAIN_FILE_NAME, 'rb'))
    x_test, y_test, y_test_label = pickle.load(open(TEST_FILE_NAME, 'rb'))

    # Dictionaries
    saved_dict = pickle.load(open(f'{PARENT_PATH}/saved_dict.pkl', 'rb'))
    mode_dict = pickle.load(open(f'{PARENT_PATH}/mode_dict.pkl', 'rb'))

    # Standard scaler
    scaler = pickle.load(open(f'{PARENT_PATH}/scaler.pkl', 'rb'))

    # Onehot/Label encoders
    ohe_dtos = pickle.load(open(f'{PARENT_PATH}/ohe_dtos.pkl', 'rb'))
    ohe_stos = pickle.load(open(f'{PARENT_PATH}/ohe_stos.pkl', 'rb'))
    ohe_dir = pickle.load(open(f'{PARENT_PATH}/ohe_dir.pkl', 'rb'))
    ohe_proto = pickle.load(open(f'{PARENT_PATH}/ohe_proto.pkl', 'rb'))

    # Making the train data sparse matrix
    x_train_csr = csr_matrix(x_train.values)

    # Creating sparse dataframe with x_train sparse matrix
    x_train = pd.DataFrame.sparse.from_spmatrix(x_train_csr, columns=x_train.columns)

    # Making test data sparse matrix
    x_test_csr = csr_matrix(x_test.values)

    # Creating x_test sparse dataframe
    x_test = pd.DataFrame.sparse.from_spmatrix(x_test_csr, columns=x_test.columns)

    return x_train_csr, y_train, x_test_csr, y_test


def train_cpu_random_forest(x_train, y_train):
    # Initialize the CPU-based Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=32, max_depth=16, random_state=42, verbose=1, n_jobs=-1)

    # Train the classifier
    rf_classifier.fit(x_train, y_train)

    return rf_classifier

if __name__ == '__main__':
    result_dict = {}

    print('-----Load data and preprocessed files------')
    x_train_csr, y_train, x_test_csr, y_test = loadDataAndPrepFiles()

    print('-----Train random forest model------')
    rf_model = train_cpu_random_forest(x_train_csr, y_train)

    # Optionally, save the trained model
    with open('rf_model.pkl', 'wb') as file:
        pickle.dump(rf_model, file)

    print('-----Evaluate random forest model------')
    # load the trained model
    with open('rf_model.pkl', 'rb') as file:
        rf_model = pickle.load(file)

    rf_clf, rf_auc, rf_f1, rf_far = evaluate_result(rf_model, x_train_csr, y_train, x_test_csr, y_test, 'RF', 'Confusion matrix of Random Forest', f'{FIGURE_PATH}/Fig1.png')

    print("Results: ")
    print(f"Random Forest: AUC = {rf_auc}, F1 = {rf_f1}, FAR = {rf_far}")