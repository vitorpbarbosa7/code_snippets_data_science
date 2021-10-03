import numpy as np
import optuna  # pip install optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
import lightgbm as lgbm

class Tuning():
    
    def __init__(self, df, target_var:str,
                n_trials:int = 10,
                chosen_optimization:str = 'recall'):

        self.df = df
        self.n_trials = n_trials    
        self.target_var = target_var
        self.chosen_optimization = chosen_optimization

        self.opts()

        self.study = optuna.create_study(direction=self.direction)
        self.func = lambda trial: self.objective(trial)
        self.study.optimize(self.func, n_trials=self.n_trials)

    def opts(self):
        maxmin = {
            "recall_overfitting": "minimize",
            "recall": "maximize",
            "f1_score": "maximize"  
        }
        self.direction = maxmin.get(self.chosen_optimization)

        # chosen_optimization
        evals = {
            "recall_overfitting": "binary_logloss",
            "recall": "average_precision",
            "f1_score": "average_precision"
        }
        self.eval_metric = evals.get(self.chosen_optimization)

    def objective(self, trial):
        param_grid = {
            #
            'boosting': trial.suggest_categorical("boosting", ['dart','rf']),
            "device_type": trial.suggest_categorical("device_type", ['cpu']),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 1.0, 1.5, step = 0.1),
            "n_estimators": trial.suggest_int("n_estimators", 50,200, step = 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, step = 0.01),
            "num_leaves": trial.suggest_int("num_leaves", 5, 40, step=5),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 150, step=10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-5, 1e-3, step = 1e-4),
            "max_bin": trial.suggest_int("max_bin", 100, 400),
            "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
            "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", 0.2, 0.95, step=0.1
            ),
            "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
            "feature_fraction": trial.suggest_float(
                "feature_fraction", 0.2, 0.95, step=0.1
            ),
        }

        # split
        X_train, X_test, y_train, y_test = self.split()

        # model fit
        model = lgbm.LGBMClassifier(objective="binary", silent = True, **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric=self.eval_metric,
            early_stopping_rounds=100,
            callbacks=[
                LightGBMPruningCallback(trial, self.eval_metric)
            ],  # Add a pruning callback
            verbose = False
        )
        # Metrics
        def recall(y_real, model_pred):
            return np.round(recall_score(y_real,model_pred),2)

        def auc(y_real, model_probs):
            '''AUC'''
            return np.round(roc_auc_score(y_real,model_probs),2)

        def precision(y_real, model_probs):
            return np.round(precision_score(y_real,model_probs),2)

        def f1(y_real, model_probs):
            return np.round(f1_score(y_real,model_probs),2)

        # probabilities
        y_train_probs = model.predict_proba(X_train)[:,1]
        y_test_probs = model.predict_proba(X_test)[:,1]

        y_train_pred = np.where(y_train_probs > 0.5 ,1 ,0)
        y_test_pred = np.where(y_test_probs > 0.5 ,1 ,0)

        # for optuna optimization
        self.recall_test = recall(y_test, y_test_pred)
        self.recall_train = recall(y_train, y_train_pred)

        self.train_auc = auc(y_real = y_train, model_probs = y_train_probs)
        self.test_auc = auc(y_real = y_test, model_probs = y_test_probs)
        
        self.precision_train = precision(y_real = y_test, model_probs = y_test_pred)
        
        self.f1_score_test = f1(y_real = y_test, model_probs = y_test_pred)

        #cv_scores[idx] = preds

        options = {
            "recall_overfitting": (self.recall_train - self.recall_test),
            "recall": self.recall_test,
            "f1_score": self.f1_score_test
        }

        return options.get(self.chosen_optimization)

    

    def split(self, test_size:float=0.3):
    
        X = self.df.drop(self.target_var, axis = 1)
        y = self.df[self.target_var]
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size = test_size,
                                                            random_state = 42)
        return X_train, X_test, y_train, y_test