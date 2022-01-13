def optimum_K()

    # Range de valores K que iremos testar

    # Utilização de valores ímpares
    KVals = range(1,30,2)

    # Lista vazia para receber a métrica
    acuracias = []

    for K in KVals:
        
        # Treinamento o modelo KNN com cada valor k

        from sklearn.neighbors import KNeighborsClassifier
        
        # Instanciar o modelo
        modeloKNN = KNeighborsClassifier(n_neighbors=K)
        
        # Realizar o treinamento 
        modeloKNN.fit(X_train, y_train)
        
        # Validar o modelo durante o treinamento com os dados de validação
        score = modeloKNN.score(X_validation, y_validation)
        
        print("Com o valor de K = %d, a acurácia é de %.2f%%" % (K, score*100))
        
        # Armazenar acuracias
        acuracias.append((K,score))

        return acuracias