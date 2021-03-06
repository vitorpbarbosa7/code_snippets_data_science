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


# class Metrics():
    
#     def __init__(self, y_real, model_probs:np.array, threshold:float = 0.5, **kwargs):
#         '''
#         y_real: pandas.Series (accepted values: {0,1,True,False}
#         model_probs: pandas.Series - probabilities generated by model
#         '''
#         y_real = y_real
#         model_probs = model_probs
#         threshold = threshold
        
#         model_pred = np.where(model_probs > threshold, 1,0)
        
#         modelname = kwargs.get('modelname',None)

#     @property
#     def charts(self):
#         '''
#         Plos the metrics charts
#         '''
#         #ks = kschart()
#         pr = precisionrecall()
#         rc = roccurve()
        
# #         return ks
    
#     def roccurve(self):
#         '''Curva roc'''
#         # Gerar os dados da diagonal (no skill classifier)
        
#         ns_model_probs = [0 for item in range(len(y_real))]
#         ns_fpr, ns_tpr, ns_thres = roc_curve(y_real, ns_model_probs)

#         #Probabilidades da classe positiva
#         fpr, tpr, thresholds = roc_curve(y_real, model_probs)
        
#         plt.plot(fpr, tpr, marker = '.', label = modelname)
#         plt.plot(ns_fpr, ns_tpr, linestyle = '--', label = 'Classificador base')
#         plt.xlabel('Raz??o de Falsos Positivos')
#         plt.ylabel('Raz??o de Verdadeiros Positivos')
#         plt.title('ROC Curve')
#         plt.legend()
#         plt.show()

#     def precisionrecall(self):
#         '''Precision Recall Curve'''

#         precision, recall, thresholds = precision_recall_curve(y_real, model_probs)
#         no_skill = len(y_real[y_real ==1]) / len(y_real)

#         plt.plot([0,1], [no_skill, no_skill], linestyle = '--', label = 'Classificador base')
#         plt.plot(precision, recall, marker = '.', label =  modelname)
#         plt.xlabel('Recall')
#         plt.ylabel('Precision')
#         plt.title('Precision-Recall Curve')
#         plt.legend()
#         plt.show()
        


#     @property 
#     def scores(self):
#         scores = pd.DataFrame({'accuracy':[accuracy()],
#                                   'precision':[precision()],
#                                   'recall':[recall()],
#                                   'f1':[f1()],
#                                   'auc':[auc()],
#                                   'gini':[gini()]})

#         return scores

#     def cvresults(self, cv = 5):
#         '''Metricas gerais: 
#         - acuracia
#         - f1
#         - recall
#         - precision'''
#         from sklearn.model_selection import cross_validate
#         res = cross_validate(model, X, y, cv = cv, 
#                                   scoring = ['accuracy','f1','recall','precision'],
#                                   return_train_score = True)

#         f1 = res['test_f1'].mean()
#         precision = res['test_precision'].mean()
#         recall = res['test_recall'].mean()
#         acuracia = res['test_accuracy'].mean()

#         cvresults = pd.DataFrame({'acuracia':[acuracia],
#                                   'precisao':[precision],
#                                   'recall':[recall],
#                                   'f1':[f1]})

#         return cvresults


#     def matriz(self):
#         from libs.confusionmatrix import matriz_confusao as cm
#         cm = cm(y_test, pred())
#         return cm