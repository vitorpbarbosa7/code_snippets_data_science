class matriz_confusao():

    import pandas as pd
    import numpy as np 
    import matplotlib.pyplot as plt

    def __init__(self, y_true, y_pred, multiclass):

        # com self.variable_endogena - variable_exogena defino que minha variável interna
        # é igual à minha variável externa

        self.y_true = y_true
        self.y_pred = y_pred
        self.multiclass = multiclass
        
        # Execução logo no início. 
        # Poderia utilizar main ou criar alguma função, por exemplo execute()
        # -------------------------------------
        # no labels eu só preciso passar o self
        matriz, lbl, mimg = self.labels()

        self.image(matriz, lbl, mimg)

    def labels(self):

        matriz = confusion_matrix(y_true = self.y_true, y_pred = self.y_pred);

        group_counts = ["{0:0.0f}".format(value) for value in matriz.flatten()]

        group_percentages = ["{0:.2%}".format(value) for value in matriz.flatten()/np.sum(matriz)]

        if self.multiclass:
            lbl = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]
            mimg = 2 # Multiplicador da imagem
        else:
            group_names = ['Verdadeiro Negativo','Falso Positivo','Falso Negativo','Verdadeiro Positivo']
            lbl = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
            mimg = 1 # Multiplicador da imagem

        return matriz, lbl, mimg

    def image(self, matriz, lbl, mimg):

        import seaborn as sns 

        shape = pd.DataFrame(self.y_true).value_counts().shape[0]
        lbl = np.asarray(lbl).reshape(shape,shape)
        plt.figure(figsize = (5*mimg,4*mimg), dpi = 150)

        from sklearn.metrics import accuracy_score
        print(f'Acurácia do modelo nos dados de validação: ({np.round(accuracy_score(self.y_true, self.y_pred),4)}) \n')
        print('Matriz de confusão:')
        sns.heatmap(matriz, annot=lbl, fmt='', cmap='Blues');
                


