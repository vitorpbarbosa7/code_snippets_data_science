import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.metrics import roc_curve, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_validate 

from .confusionmatrix import MatrizConfusao
# from libs.kolgo import KolgomorovSmirnov

class Metrics():
    
    def __init__(self, y_real, model_probs:np.array, threshold:float = 0.5, **kwargs):
        '''
        y_real: pandas.Series (accepted values: {0,1,True,False}
        model_probs: pandas.Series - probabilities generated by model
        '''
        self.y_real = y_real
        self.model_probs = model_probs
        self.threshold = threshold
        
        self.model_pred = np.where(self.model_probs > self.threshold, 1,0)
        
        self.modelname = kwargs.get('modelname',None)

    @property
    def charts(self):
        '''
        Plos the metrics charts
        '''
        #self.ks = self.kschart()
        self.pr = self.precisionrecall()
        self.rc = self.roccurve()
        
#         return self.ks
    
    def roccurve(self):
        '''Curva roc'''
        # Gerar os dados da diagonal (no skill classifier)
        
        ns_model_probs = [0 for item in range(len(self.y_real))]
        ns_fpr, ns_tpr, ns_thres = roc_curve(self.y_real, ns_model_probs)

        #Probabilidades da classe positiva
        fpr, tpr, thresholds = roc_curve(self.y_real, self.model_probs)
        
        plt.plot(fpr, tpr, marker = '.', label = self.modelname)
        plt.plot(ns_fpr, ns_tpr, linestyle = '--', label = 'Classificador base')
        plt.xlabel('Razão de Falsos Positivos')
        plt.ylabel('Razão de Verdadeiros Positivos')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

    def precisionrecall(self):
        '''Precision Recall Curve'''

        precision, recall, thresholds = precision_recall_curve(self.y_real, self.model_probs)
        no_skill = len(self.y_real[self.y_real ==1]) / len(self.y_real)

        plt.plot([0,1], [no_skill, no_skill], linestyle = '--', label = 'Classificador base')
        plt.plot(precision, recall, marker = '.', label =  self.modelname)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()
        
    def accuracy(self):
        return np.round(accuracy_score(self.y_real, self.model_pred),2)

    def precision(self):
        return np.round(precision_score(self.y_real, self.model_pred),2)

    def recall(self):
        return np.round(recall_score(self.y_real, self.model_pred),2)

    def f1(self):
        return np.round(f1_score(self.y_real, self.model_pred),2)

    def auc(self):
        '''AUC'''
        return np.round(roc_auc_score(self.y_real, self.model_probs),2)
    
    def gini(self):
        '''GINI'''
        return 2*self.auc() - 1

    def avgprec(self):
        '''Average Precision Score'''
        return np.round(average_precision_score(self.y_real, self.model_probs),2)

    @property
    def scores(self):
        scores = pd.DataFrame({'accuracy':[self.accuracy()],
                                  'precision':[self.precision()],
                                  'recall':[self.recall()],
                                  'f1':[self.f1()],
                                  'auc':[self.auc()],
                                  'gini':[self.gini()]})

        return scores

    def cvresults(self, cv = 5):
        '''Metricas gerais: 
        - acuracia
        - f1
        - recall
        - precision'''
        res = cross_validate(self.model, self.X, self.y, cv = cv, 
                                  scoring = ['accuracy','f1','recall','precision'],
                                  return_train_score = True)

        f1 = res['test_f1'].mean()
        precision = res['test_precision'].mean()
        recall = res['test_recall'].mean()
        acuracia = res['test_accuracy'].mean()

        cvresults = pd.DataFrame({'acuracia':[acuracia],
                                  'precisao':[precision],
                                  'recall':[recall],
                                  'f1':[f1]})

        return cvresults


    def matriz(self):
        cm = MatrizConfusao(self.y_test, self.pred())
        return cm