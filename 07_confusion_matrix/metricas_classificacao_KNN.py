class metricas_classificacao_KNN():

	def __init__(self, X_train, y_train, X_valid, y_valid):

		self.X_train = X_train
		self.y_train = y_train
		self.X_valid = X_valid
		self.y_valid = y_valid

		# Variáveis que foram declaradas como self já estão disponíveis para 
		# outras funções desta classe

		# Execução:
		self.optimum_K(X_train, y_train, X_valid, y_valid)

	def optimum_K(self, X_train, y_train, X_valid, y_valid):
    
	    import numpy as np
	    from sklearn.neighbors import KNeighborsClassifier

	    # Range de valores K que iremos testar

	    # Utilização de valores ímpares
	    KVals = range(1,30,2)

	    # Lista vazia para receber a métrica
	    metricas = []
	    
	    for K in KVals:
	        
	        # Treinamento o modelo KNN com cada valor k

	        # Instanciar o modelo
	        modeloKNN = KNeighborsClassifier(n_neighbors=K)
	        
	        # Realizar o treinamento 
	        modeloKNN.fit(self.X_train, self.y_train)
	        
	        # Validar o modelo durante o treinamento com os dados de validação
	        score = modeloKNN.score(self.X_valid, self.y_valid)
	        
	        y_pred = modeloKNN.predict(self.X_valid)
	        
	        accuracy, recall, precision, f1score = self.ac_pr_re_f1(y_true = self.y_valid,y_pred = y_pred)
	        
	        print("Com o valor de K = %d, a acurácia é de %.2f%%, a precision de %.2f%%, a recall de %.2f%%, e o F1-Score de %.2f%%" % (K, score*100, precision*100, recall*100, f1score*100))
	        
	        # Armazenar acuracias
	        metricas.append(np.round((K,accuracy,recall,precision,f1score),3))

	    return metricas
    
    # A função que calcula acurácia, precision, recall e f1score
	def ac_pr_re_f1(self, y_true, y_pred):
	    
	    from sklearn.metrics import confusion_matrix, accuracy_score
	    import numpy as np
	    import matplotlib.pyplot as plt
	    import seaborn as sns
	    
	    matriz = confusion_matrix(y_true = y_true, y_pred = y_pred);
	    TN, FP, FN, TP = matriz.flatten()
	    
	    # Acurácia
	    accuracy = (TP + TN)/(TP + TN + FP + FN)
	    
	    # Precision
	    precision = TP/(TP + FP)
	    
	    # Recall
	    recall = TN/(TN + FN)
	    
	    # F1-Score
	    f1score = 2*(precision*recall)/(precision + recall)
	    
	    return accuracy, precision, recall, f1score