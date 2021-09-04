
from sklearn.metrics import confusion_matrix
import numpy as np 

class matriz_confusao():

    def __init__(self, y_test, y_pred):

        # com self.variable_endogena - variable_exogena defino que minha variável interna
        # é igual à minha variável externa

        self.y_test = y_test
        self.y_pred = y_pred

        # Execução logo no início. 
        # Poderia utilizar main ou criar alguma função, por exemplo execute()
        # -------------------------------------
        # no labels eu só preciso passar o self
        self.matriz, self.lbl, self.mimg = self.labels()

        
        self.image()

    def return_values(self):
        return confusion_matrix(y_true = self.y_test, y_pred = self.y_pred).ravel()

    def return_pcts(self):

        matriz = confusion_matrix(y_true = self.y_test, y_pred = self.y_pred)

        group_counts = ["{0:0.0f}".format(value) for value in matriz.flatten()]

        group_percentages = ["{0:.2%}".format(value) for value in matriz.flatten()/np.sum(matriz)]
        
        return group_percentages

    def labels(self):

        matriz = confusion_matrix(y_true = self.y_test, y_pred = self.y_pred)

        group_counts = ["{0:0.0f}".format(value) for value in matriz.flatten()]

        group_percentages = ["{0:.2%}".format(value) for value in matriz.flatten()/np.sum(matriz)]

        group_names = ['Verdadeiro Negativo','Falso Positivo','Falso Negativo','Verdadeiro Positivo']
        lbl = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
        mimg = 1 # Multiplicador da imagem

        return matriz, lbl, mimg

    def image(self):

        import seaborn as sns 
        import pandas as pd 
        import matplotlib.pyplot as plt
        import numpy as np

        df = pd.DataFrame(self.y_test)
        shape = df.value_counts().shape[0]
        lbl_ = np.asarray(self.lbl).reshape(shape,shape)
        plt.figure(figsize = (5*self.mimg,4*self.mimg), dpi = 150)

        print('Matriz de confusão:')
        sns.heatmap(self.matriz, annot=lbl_, fmt='', cmap='Blues');