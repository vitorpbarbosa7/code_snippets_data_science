import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.metrics import roc_curve, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_validate 
# from libs.kolgo import KolgomorovSmirnov

from abc import ABC, abstractmethod

# parameters
_houses = 2

def _round(_value):
    return np.round(_value,_houses)

class GeneralMetrics(ABC):
    ''' Any metrics'''
    @abstractmethod
    def metric(y_real, model_pred):
        pass

class accuracy(GeneralMetrics):
    def metric(y_real, model_pred):
        return pd.Series({'accuracy':f'{_round(accuracy_score(y_real, model_pred))}'})

class precision(GeneralMetrics):
    def metric(y_real, model_pred):
        return pd.Series({'precision':f'{_round(precision_score(y_real, model_pred))}'})



class Metrics:
    '''Main'''
    @abstractmethod
    def get_metrics(y_real, model_pred):

        return print(pd.concat(metric.metric(y_real, model_pred) for metric in GeneralMetrics.__subclasses__()))

        for metric in GeneralMetrics.__subclasses__():
            # _dataframe = pd.DataFrame()
            # _dataframe.append(
            metric.metric(y_real, model_pred)

if __name__ == '__main__':

    y_real = [1,0,0,0,1,0,1,0,1]
    model_pred = [1,0,0,0,1,1,1,0,1]
    Metrics.get_metrics(y_real, model_pred)