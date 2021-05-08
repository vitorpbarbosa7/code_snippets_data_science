def matriz_confusao(y_true, y_pred, multiclass = False):

    import pandas as pd 
    import numpy as np 
    from sklearn.metrics import accuracy_score
    import seaborn as sns 
    import matplotlib.pyplot as plt

    matriz = confusion_matrix(y_true = y_true, y_pred = y_pred);

    group_counts = ["{0:0.0f}".format(value) for value in matriz.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in matriz.flatten()/np.sum(matriz)]
    
    if multiclass:
        labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]
        mimg = 2 # Multiplicador da imagem
    else:
        group_names = ['Verdadeiro Negativo','Falso Positivo','Falso Negativo','Verdadeiro Positivo']
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
        mimg = 1 # Multiplicador da imagem

    shape = pd.DataFrame(y_true).value_counts().shape[0]
    labels = np.asarray(labels).reshape(shape,shape)
    plt.figure(figsize = (5*mimg,4*mimg), dpi = 150)
    
    print(f'Acurácia do modelo nos dados de validação: ({np.round(accuracy_score(y_true, y_pred),4)}) \n')
    print('Matriz de confusão:')
    sns.heatmap(matriz, annot=labels, fmt='', cmap='Blues');