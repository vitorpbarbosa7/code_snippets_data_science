# Printar matriz de confusão
def model_pipeline(X_train, X_test, y_train, y_test, threshold):
    
    # Pacotes
    import numpy as np 
    import pandas as pd
    import statsmodels.api as sm
    import binary_classification as bc
    
    # Fit do modelo
    modelo = sm.Logit(exog = X_train ,endog = y_train.values).fit()

    # Printar resultado do modelo    
    print(modelo.summary())

    # Obter as probabilidades
    pred_prob = modelo.predict(X_test)

    # Transformar para binário de acordo com o thresnold
    pred_binary = np.where(pred_prob > threshold, 1, 0)
    
    # Printar a matriz de confusão
    bc.matriz_confusao(y_true = y_test, y_pred = pred_binary)
    
    # Plotar curva ROC
    bc.roccurve(y_test = y_test, probs = pred_prob, modelname = 'Logistic Regression')

    # Plotar curva Precision- Recall
    bc.precisionrecall(y_test = y_test, probs = pred_prob, modelname = 'Logistic Regression')

def matriz_confusao(y_true, y_pred):
    
    from sklearn.metrics import confusion_matrix, accuracy_score
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    matriz = confusion_matrix(y_true = y_true, y_pred = y_pred);
    TN, FP, FN, TP = matriz.flatten()
    
    # Precision
    precision = TP/(TP + FP)
    
    # Recall
    recall = TN/(TN + FN)
    
    # F1-Score
    f1score = 2*(precision*recall)/(precision + recall)
    
    group_names = ['Verdadeiro Negativo','Falso Positivo','Falso Negativo','Verdadeiro Positivo']

    group_counts = ["{0:0.0f}".format(value) for value in matriz.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in matriz.flatten() / np.sum(matriz)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)
    plt.figure(figsize = (5,4), dpi = 130)
    
    print(f'Acurácia do modelo nos dados de validação: ({np.round(accuracy_score(y_true, y_pred),4)}) \n')
    
    print(f'Precision: ({np.round(precision,4)}) \n')
    print(f'Recall: ({np.round(recall,4)}) \n')
    print(f'F1-Score: ({np.round(f1score,4)}) \n')
    
    print('Matriz de confusão:')
    sns.heatmap(matriz, annot=labels, fmt='', cmap='Blues');
    plt.show()


# Printar curva ROC
def roccurve(y_test, probs, modelname):
    
    from sklearn.metrics import roc_curve, roc_auc_score
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Gerar os dados da diagonal (no skill classifier)
    ns_probs = [0 for item in range(len(y_test))]
    ns_fpr, ns_tpr, ns_thres = roc_curve(y_test, ns_probs)
    
    #Probabilidades da classe positiva
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    
    plt.figure(figsize = (5,4), dpi = 130)
    plt.plot(fpr, tpr, marker = '.', label = modelname)
    plt.plot(ns_fpr, ns_tpr, linestyle = '--', label = 'Classificador base')
    plt.xlabel('Razão de Falsos Positivos')
    plt.ylabel('Razão de Verdadeiros Positivos')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

# Printar Precision-Recall
def precisionrecall(y_test, probs, modelname):

    from sklearn.metrics import precision_recall_curve
    import numpy as np
    import matplotlib.pyplot as plt
    
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    
    no_skill = len(y_test[y_test ==1]) / len(y_test)
    
    plt.figure(figsize = (5,4), dpi = 130)
    plt.plot([np.min(recall),1], [no_skill, no_skill], linestyle = '--', label = 'Classificador base')
    plt.plot(recall, precision, marker = '.', label =  modelname)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

    # Figura apenas com o threshold e recall 

    plt.figure(figsize = (5,4), dpi = 130)
    plt.scatter(x = thresholds, y = recall[:-1]) # Recall size is thresholds.shape + 1
    plt.xlabel('Threshold')
    plt.ylabel('Recall')
    plt.show()