import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import json
import re
import pickle

from libs.pipeline import Pipe

df = pd.read_csv('data/UCI_Credit_Card_renamed.csv')

with open('bestparams.json') as file:
    bestparams = json.load(file)

print(bestparams)

model = lgb.LGBMClassifier().set_params(**bestparams)

def split(df, target_var:str, test_size:float=0.3):

    X = df.drop(target_var, axis = 1)
    y = df[target_var]
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size = test_size,
                                                        random_state = 42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split(df, target_var = 'target')
df_train = pd.concat([X_train, y_train], axis = 1)
df_test = pd.concat([X_test, y_test], axis = 1)
df_train.to_csv('data/df_train.csv', index = False)
df_test.to_csv('data/df_test.csv', index = False)

model = Pipe(df_train, target_var = 'target', model = model)
print(model.popin.scores)
print(model.oos.scores)
lgbm_model = model.model
print(lgbm_model)

with open('lgbm_model.pickle', 'wb') as file:
    pickle.dump(lgbm_model, file)
