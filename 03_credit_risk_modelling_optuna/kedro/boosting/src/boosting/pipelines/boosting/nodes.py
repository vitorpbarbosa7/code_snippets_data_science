"""
This is a boilerplate pipeline 'boosting'
generated using Kedro 0.18.5
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
import scipy.stats as stats
import statsmodels.stats.api as sms
import optuna
from sklearn.model_selection import train_test_split
from .libs.tuning_cv import Tuning
from .libs.evaluation import Metrics
from plotnine import *

import lightgbm as lgb

def data_prep(df:pd.DataFrame):

    df.columns = df.columns.str.lower()

    df = df.drop('id', axis = 1)

    df = df.rename(columns = {'default.payment.next.month':'target'})

    return df


def split(df:pd.DataFrame):

    target_var = 'target'
    X = df.drop(target_var, axis = 1)
    y = df[target_var]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    df_train = pd.concat([X_train, y_train], axis = 1)
    df_test = pd.concat([X_test, y_test], axis = 1)

    df_train = df_train.reset_index(drop=True)

    df_test = df_test.reset_index(drop=True)

    return df_train, df_test

def opt(df_train:pd.DataFrame,
            optuna_metric):

    tuning = Tuning(df_train,
                target_var = 'target', 
                n_trials = 10,
                metric='average_precision',
                direction = 'maximize',
                optuna_metric = optuna_metric)

    tuning.exec()

    optuna_study = tuning.study

    bestparams = optuna_study.best_params

    return optuna_study, bestparams

def train(df_train, bestparams):

    model = lgb.LGBMClassifier(**bestparams)

    X_train = df_train.drop('target', axis = 1)
    y_train = df_train['target']

    model.fit(X = X_train, y = y_train)

    return model

def prediction(df_test:pd.DataFrame, model):

    X_test = df_test.drop('target', axis = 1)
    y_test = pd.Series(df_test['target'])

    y_hat = model.predict_proba(X_test)[:,1]

    y_test = pd.DataFrame({'y_test':y_test})
    y_hat = pd.DataFrame({'y_hat':y_hat})

    return y_hat, y_test

def evaluate(y_hat, y_test):

    y_hat = y_hat['y_hat'].values
    y_test = y_test['y_test'].values

    metrics = Metrics(y_real = y_test, model_probs = y_hat)

    scores = metrics.scores

    return scores

def plot(y_hat, y_test):

    y_hat = y_hat['y_hat'].values
    y_test = y_test['y_test'].values

    results = pd.DataFrame({'y_test':y_test,'y_hat':y_hat})
    results['y_test'] = results['y_test'].astype('category')

    fig, plot = (
        ggplot(data = results, mapping = aes(x = 'y_hat')) + 
        geom_density(aes(color = 'y_test'))
    ).draw(show = False, return_ggplot=True)

    return fig


    





    








