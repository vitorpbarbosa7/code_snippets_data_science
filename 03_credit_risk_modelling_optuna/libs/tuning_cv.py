import pandas as pd
import numpy as np
import optuna  # pip install optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import lightgbm as lgbm

import logging

log = logging.getLogger(__name__)

class Tuning():
    
    def __init__(self, df:pd.DataFrame, 
                 target_var:str,
                 n_trials:int = 10,
                 metric:str = 'average_precision',
                 direction:str = 'maximize'):
        
        '''
        Parameters
        ---------------
        df: Dataframe containing exp vars and target vars 
        target_var: target variable for classification
        n_trials: number of times optuna will attempt to change hyperparameter values to reach better final score
        metric:{'recall_overfitting', 'recall', 'f1_score'}
            - recall_overfitting: optiom minimizes the difference (recall_train - recall_test)
            - recall: maximizes recall_test
            - f1_score: maximizes f1_score test
        '''
        self.df = df
        self.n_trials = n_trials    
        self.target_var = target_var
        self.metric = metric
        self.direction = direction


    def exec(self):
        self.study = optuna.create_study(direction=self.direction)
        self.func = lambda trial: self.objective(trial)
        self.study.optimize(self.func, n_trials=self.n_trials)

        return self.study


    def objective(self, trial):
        param_grid = {
            'objective': 'binary',
            'metric': self.metric,
            'boosting': trial.suggest_categorical("boosting", ['dart']),
            "n_estimators": trial.suggest_int("n_estimators", 50,200, step = 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, step = 0.01),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "num_leaves": trial.suggest_int("num_leaves", 5, 40, step=5),
            "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
            "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 0.7, step=0.1),
            "bagging_freq": trial.suggest_int("bagging_freq", 1,20, step = 5),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.5, step=0.1)
        }
        
        # X_train, X_test, y_train, y_test = self.split()

        # model is created with a param_grid for optuna
        model = lgbm.LGBMClassifier(silent = True, 
                                    **param_grid)

        X = self.df.drop(self.target_var, axis = 1)
        y = self.df[self.target_var]

        kf = KFold(n_splits=3, shuffle=True, random_state=0)
        train_scores = []
        test_scores = []
        count = 1
        for train_index, test_index in kf.split(X):
            log.info(f'Cross validation - {count}')
            print(f'Cross validation - {count}')
            X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
            y_train , y_test = y[train_index] , y[test_index]

            y_train = y_train.values
            y_test = y_test.values

            # step of this objective function for optuna, in which the param_grids will already be decided by optuna, which one try this time
            eval_metric = 'binary_logloss'
            pruning_callback = LightGBMPruningCallback(trial, eval_metric)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                # eval_metric=eval_metric,
                # early_stopping_rounds=100,
                # callbacks=[pruning_callback],
                verbose = False
            )

            # probabilities
            y_train_probs = model.predict_proba(X_train)[:,1]
            y_test_probs = model.predict_proba(X_test)[:,1]
            y_train_pred = np.where(y_train_probs > 0.5 ,1 ,0)
            y_test_pred = np.where(y_test_probs > 0.5 ,1 ,0)

            # Metrics
            # Possitibility to create custom metrics
            # self.recall_test = self.recall(y_test, y_test_pred)
            # self.recall_train = self.recall(y_train, y_train_pred)

            # self.train_auc = self.auc(y_real = y_train, model_probs = y_train_probs)
            # self.test_auc = self.auc(y_real = y_test, model_probs = y_test_probs)
            
            # self.precision_train = self.precision(y_real = y_test, model_probs = y_test_pred)
            
            print('Calculating optuna final objective metrics')
            print(y_train)
            self.f1_score_train = f1_score(y_train, y_train_pred)
            self.f1_score_test = f1_score(y_test, y_test_pred)

            train_score = self.f1_score_train
            test_score = self.f1_score_test

            train_scores.append(train_score)
            test_scores.append(test_score)

            count += 1

            print('final first cross validation')

        # final average metric after Cross Validation
        final_metric_test = np.mean(test_scores)

        return final_metric_test

    def recall(y_real, model_pred):
        return np.round(recall_score(y_real,model_pred),2)

    def auc(y_real, model_probs):
        '''AUC'''
        return np.round(roc_auc_score(y_real,model_probs),2)

    def precision(y_real, model_probs):
        return np.round(precision_score(y_real,model_probs),2)

    def f1(y_real, model_probs):
        return np.round(f1_score(y_real,model_probs),2)

    def split(self, test_size:float=0.3):
    
        X = self.df.drop(self.target_var, axis = 1)
        y = self.df[self.target_var]
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size = test_size,
                                                            random_state = 42)
        return X_train, X_test, y_train, y_test