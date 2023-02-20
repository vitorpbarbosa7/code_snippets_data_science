import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import optuna
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import json
import re

from libs.tuning import Tuning

df = pd.read_csv("data/UCI_Credit_Card.csv")
df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
df.columns = df.columns.str.lower()
df = df.drop('id', axis = 1)
df = df.rename(columns = {'defaultpaymentnextmonth':'target'})
df.to_csv("data/UCI_Credit_Card_renamed.csv", index = False)

# Optuna
# split    

def split(df, target_var:str, test_size:float=0.3):

    X = df.drop(target_var, axis = 1)
    y = df[target_var]
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size = test_size,
                                                        random_state = 42)
    return X_train, X_test, y_train, y_test

tuning = Tuning(df,
                target_var = 'target', 
                n_trials = 30,
                chosen_optimization='f1_score')

bestparams = tuning.study.best_params
print(bestparams)

with open('bestparams.json', 'w') as file:
    json.dump(bestparams, file)


