import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import lightgbm as lgb
import re
import pickle

from sklearn.metrics import precision_recall_curve

with open('lgbm_model.pickle', 'rb') as file:
    lgbm_model = pickle.load(file)

df_test = pd.read_csv('data/df_test.csv')
X_test = df_test.drop('target', axis = 1)
y_test = df_test['target']

y_hat = lgbm_model.predict_proba(X_test)[:,1]

precision, recall, thresholds = precision_recall_curve(y_test, y_hat)

print(len(precision), len(recall), len(thresholds))

df_cuts = pd.DataFrame({'precision':precision[1:],'recall':recall[1:],'thresholds': thresholds})

print(df_cuts)

no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0,1], [no_skill, no_skill], linestyle = '--', label = 'Classificador base')
plt.plot(precision, recall, marker = '.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
