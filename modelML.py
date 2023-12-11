'''
Written by Yuening (Lily) Li and Haoli Yin
'''

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import os
import pickle
import datetime
import itertools
from xgboost import XGBClassifier

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
PREPROCESSED_PATH = '../data'

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

    train_filt = x_train.isna().any(axis=1)
    if train_filt.sum() > 0:
        print(f"filtered out {train_filt.sum()} records containing NaNs in training set")
        x_train = x_train.loc[~train_filt, :]
        y_train = y_train.loc[~train_filt]
    # Making train data sparse matrix
    x_train_csr = csr_matrix(x_train.values)

    test_filt = x_test.isna().any(axis=1)
    if test_filt.sum() > 0:
        print(f"filtered out {train_filt.sum()} records containing NaNs in testing set")
        x_test = x_test.loc[~test_filt, :]
        y_test = y_train.loc[~test_filt]
    # Making test data sparse matrix
    x_test_csr = csr_matrix(x_test.values)

    return x_train_csr, y_train, x_test_csr, y_test




def train_cpu_random_forest(x_train, y_train):
    # Initialize the CPU-based Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=32, max_depth=16, random_state=42, verbose=1, n_jobs=-1)

    # Train the classifier
    rf_classifier.fit(x_train, y_train)

    return rf_classifier


def train_xgboost(x_train, y_train):

    xgb_classifier = XGBClassifier(
        n_estimators=15, max_depth=16, learning_rate=0.1,
        objective='binary:logistic'
    )
    xgb_classifier.fit(x_train, y_train)
    # make prediction
    # preds = xgb_classifier.predict(X_test)

    return xgb_classifier


if __name__ == '__main__':
    # configuration for the experiment (which model to use)
    config_dict = {
        "model_name": "rf",  # options: xgboost | rf 
        "load_model": False,  # If False, retrain model
    }
    model_name = config_dict["model_name"]
    load_model = config_dict["load_model"]

    result_dict = {}

    print('-----Load data and preprocessed files------')
    x_train_csr, y_train, x_test_csr, y_test = loadDataAndPrepFiles()

    print("x_train_csr.shape = ", x_train_csr.shape)
    print("y_train.shape = ", y_train.shape)

    if model_name in ["rf", "Random Forests"]:
        if not load_model:
            print('-----Train random forest model------')
            rf_model = train_cpu_random_forest(x_train_csr, y_train)

            # Optionally, save the trained model
            with open('rf_model.pkl', 'wb') as file:
                pickle.dump(rf_model, file)

        # load the trained model
        with open('rf_model.pkl', 'rb') as file:
            rf_model = pickle.load(file)
        the_model = rf_model

    elif model_name in ["xgboost", "XGBoost"]:  # xgboost model
        if not load_model:
            print('-----Train xgboost model------')
            xgb_model = train_xgboost(x_train_csr, y_train)

            # Optionally, save the trained model
            with open('xgb_model.pkl', 'wb') as file:
                pickle.dump(xgb_model, file)
        else:
            print('-----Load xgboost model------')
            with open('xgb_model.pkl', 'rb') as file:
                xgb_model = pickle.load(file)
        the_model = xgb_model
    else:
        raise IOError(f"unknown model name: {model_name}.")

    print(f'-----Evaluate {model_name} model------')
    rf_clf, rf_auc, rf_f1, rf_far = evaluate_result(
        the_model, x_train_csr, y_train, x_test_csr, y_test,
        model_name,
        f'Confusion matrix of {model_name}',
        f'{FIGURE_PATH}/Fig1_{model_name}.png'
    )

    print("Results: ")
    print(f"{model_name}: AUC = {rf_auc}, F1 = {rf_f1}, FAR = {rf_far}")

