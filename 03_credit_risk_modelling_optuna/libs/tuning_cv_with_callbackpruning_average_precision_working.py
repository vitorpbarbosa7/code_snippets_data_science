# Boilerplate

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
import scipy.stats as stats
import statsmodels.stats.api as sms
import json
import optuna

from IPython.display import display
from IPython.display import Markdown as md
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

def f():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    
def nf():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 4)
nf()
import warnings
warnings.filterwarnings('ignore')

def dp(df, r = 5, c = None):
    from IPython.display import display
    with pd.option_context('display.max_rows', 4, 'display.max_columns', None):
        display(df)

def fg(w = 6, h = 4, dpi = 120):
    plt.rcParams['figure.figsize'] = (w,h)
    plt.rcParams['figure.dpi'] = dpi
fg()

from libs.pipeline import Pipe
import lightgbm as lgb

df = pd.read_csv("data/UCI_Credit_Card.csv")

df.columns = df.columns.str.lower()

df = df.drop('id', axis = 1)

df = df.rename(columns = {'default.payment.next.month':'target'})

df.to_parquet('data/df_to_feature_selection.parquet', index = False)

from sklearn.model_selection import KFold

target_var = 'target'
X = df.drop(target_var, axis = 1)
y = df[target_var]

kf = KFold(n_splits=3, shuffle=True, random_state=0)
train_scores = []
test_scores = []
for train_index, test_index in kf.split(X):
    X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]

# Optuna

from libs.tuning_cv_dois import Tuning

tuning = Tuning(df,
                target_var = 'target', 
                n_trials = 2,
                metric='average_precision',
                direction = 'maximize')

tuning.exec()

optuna.visualization.plot_intermediate_values(tuning.study)

bestparams = tuning.study.best_params

bestparams

%load_ext autoreload
%autoreload

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
import scipy.stats as stats
import statsmodels.stats.api as sms
import json
import optuna

from IPython.display import display
from IPython.display import Markdown as md
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

def f():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    
def nf():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 4)
nf()
import warnings
warnings.filterwarnings('ignore')

def dp(df, r = 5, c = None):
    from IPython.display import display
    with pd.option_context('display.max_rows', 4, 'display.max_columns', None):
        display(df)

def fg(w = 6, h = 4, dpi = 120):
    plt.rcParams['figure.figsize'] = (w,h)
    plt.rcParams['figure.dpi'] = dpi
fg()

from libs.pipeline import Pipe
import lightgbm as lgb

df = pd.read_csv("data/UCI_Credit_Card.csv")

df.columns = df.columns.str.lower()

df = df.drop('id', axis = 1)

df = df.rename(columns = {'default.payment.next.month':'target'})

df.to_parquet('data/df_to_feature_selection.parquet', index = False)

from sklearn.model_selection import KFold

target_var = 'target'
X = df.drop(target_var, axis = 1)
y = df[target_var]

kf = KFold(n_splits=3, shuffle=True, random_state=0)
train_scores = []
test_scores = []
for train_index, test_index in kf.split(X):
    X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]

# Optuna

from libs.tuning_cv_dois import Tuning

tuning = Tuning(df,
                target_var = 'target', 
                n_trials = 2,
                metric='average_precision',
                direction = 'maximize')

tuning.exec()

optuna.visualization.plot_intermediate_values(tuning.study)

bestparams = tuning.study.best_params

bestparams