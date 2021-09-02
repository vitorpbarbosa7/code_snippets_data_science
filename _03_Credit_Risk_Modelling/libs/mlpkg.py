import numpy as np 
import pandas as pd

class ml():

    '''classe para pipeline de ml'''
    def __init__(self, X_train, X_test, y_train, y_test, 
                model, modelname = '', threshold = 0.5):

        # Self variables
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.model = model
        self.modelname = modelname
        self.threshold = threshold
        
        # Train model
        self.fit()
        
    def fit(self):
        self.model.fit(self.X_train, self.y_train)
        
    def probs(self, new_X_test = None):
        
        self.new_X_test = new_X_test
        
        '''Predict proba'''
        if new_X_test is None:
            return self.model.predict_proba(self.X_test)[:,1]
        else:
            return self.model.predict_proba(self.new_X_test)[:,1]
        
    def pred(self):
        return np.where(self.probs() > self.threshold, 1, 0)

    def matriz(self):
        from libs.confusionmatrix import matriz_confusao as cm
        cm = cm(self.y_test, self.pred())
        return cm

    def roccurve(self):
        '''Curva roc'''
        from sklearn.metrics import roc_curve
        import matplotlib.pyplot as plt
        # Gerar os dados da diagonal (no skill classifier)
        
        ns_probs = [0 for item in range(len(self.y_test))]
        ns_fpr, ns_tpr, ns_thres = roc_curve(self.y_test, ns_probs)

        #Probabilidades da classe positiva
        fpr, tpr, thresholds = roc_curve(self.y_test, self.probs())
        
        plt.plot(fpr, tpr, marker = '.', label = self.modelname)
        plt.plot(ns_fpr, ns_tpr, linestyle = '--', label = 'Classificador base')
        plt.xlabel('Razão de Falsos Positivos')
        plt.ylabel('Razão de Verdadeiros Positivos')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

    def precisionrecall(self):
        '''Precision Recall Curve'''
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve

        precision, recall, thresholds = precision_recall_curve(self.y_test, self.probs())
        no_skill = len(self.y_test[self.y_test ==1]) / len(self.y_test)

        plt.plot([0,1], [no_skill, no_skill], linestyle = '--', label = 'Classificador base')
        plt.plot(precision, recall, marker = '.', label =  self.modelname)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()
        

    def accuracy_score(self):
        from sklearn.metrics import accuracy_score
        return np.round(accuracy_score(self.y_test, self.pred()),2)

    def precision_score(self):
        from sklearn.metrics import precision_score
        return np.round(precision_score(self.y_test, self.pred()),2)

    def recall_score(self):
        from sklearn.metrics import recall_score
        return np.round(recall_score(self.y_test, self.pred()),2)

    def f1_score(self):
        from sklearn.metrics import f1_score
        return np.round(f1_score(self.y_test, self.pred()),2)

    def auc(self):
        '''AUC'''
        from sklearn.metrics import roc_auc_score
        return np.round(roc_auc_score(self.y_test, self.probs()),2)
    
    def gini(self):
        '''GINI'''
        return 2*self.auc() - 1

    def avgprec(self):
        '''Average Precision Score'''
        from sklearn.metrics import average_precision_score
        return np.round(average_precision_score(self.y_test, self.probs()),2)


    def scores(self):
        scores = pd.DataFrame({'acuracia':[self.accuracy_score()],
                                  'precision':[self.precision_score()],
                                  'recall':[self.recall_score()],
                                  'f1':[self.f1_score()],
                                  'auc':[self.auc()],
                                  'gini':[self.gini()]})

        return scores

    def cvresults(self, cv = 5):
        '''Metricas gerais: 
        - acuracia
        - f1
        - recall
        - precision'''
        from sklearn.model_selection import cross_validate
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