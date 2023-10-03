import pandas as pd
import numpy as np
import optuna  # pip install optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgbm

class Tuning():
    
    def __init__(self, df:pd.DataFrame, 
                 target_var:str,
                 n_trials:int = 10,
                 metric:str = 'average_precision',
                 direction:str = 'maximize'):
        
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

        X = self.df.drop(self.target_var, axis = 1)
        y = self.df[self.target_var]

        param_grid = {
        'boosting': trial.suggest_categorical("boosting", ['dart']),
#         "n_estimators": trial.suggest_int("n_estimators", 50,200, step = 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
#         "max_depth": trial.suggest_int("max_depth", 3, 8),
#         "num_leaves": trial.suggest_int("num_leaves", 5, 40, step=5),
#         "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
#         "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
#         "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 0.7, step=0.1),
#         "bagging_freq": trial.suggest_int("bagging_freq", 1,20, step = 5),
#         "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.5, step=0.1)
        }

        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=7)

        cv_scores = np.empty(5)
        for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = lgbm.LGBMClassifier(objective="binary", **param_grid)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                
                # code with make it working with at least the StratifiedKFold and pruning with binary_logloss
                #https://towardsdatascience.com/kagglers-guide-to-lightgbm-hyperparameter-tuning-with-optuna-in-2021-ed048d9838b5

                # details about the pruning and plot 
                # https://www.kaggle.com/code/corochann/optuna-tutorial-for-hyperparameter-optimization#4.-[Advanced]-Pruning-unpromising-trials-for-more-faster-search
                eval_metric="average_precision",
                # wait at least 100 trees, after that, if no significante improvement is made on the metric we're evaluation, just stop, and go on to next
                early_stopping_rounds=100,
                callbacks=[
                    LightGBMPruningCallback(trial, "average_precision")
                ],  # Add a pruning callback
            )
            preds = model.predict_proba(X_test)
            cv_scores[idx] = log_loss(y_test, preds)

        return np.mean(cv_scores)