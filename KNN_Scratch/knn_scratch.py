class knn_scratch():
    
    
    def __init__(self, X_train, X_test, y_train, y_test, K):
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.K = K
    
    # Função para retornar as previsões
    def predict(self):
        
        # Retornar matriz de distâncias
        df_distancias = self.distancias(self.X_test, self.X_train, self.y_train)

        # Retornar previsões
        return np.array(self.previsoes(df_distancias, self.K))
        
    # Função que retorna o score
    def score(self):
        
        y_pred = self.predict()
        
        # Quantas vezes é retornado 1 ou 0, para nos retornar a acurácia
        return np.sum(np.array(y_pred)==np.array(self.y_test))/y_test.shape[0]
    
    # Função que calcula as previsões
    def previsoes(self, df_distancias, K):
    
        """ Função que retorna uma lista com todas as previsões

        Parâmetros:
        -----------------------
        df_distancias: Dataframe da matriz de distâncias dos pontos de test com os pontos de treino

        k

        """
    
        return [self.one_pred(df_distancias, K = self.K, col = ft) for ft in np.array(range(df_distancias.shape[1] - 1))]
    
    # Função que calcula uma previsão
    def one_pred(self, df_distancias, K, col):
    
        """ Função para retornar a classificação, ou seja, a predição

        Parâmetros
        ----------------
        aux: DataFrame da matriz de distâncias com colunas que correspondem as features de teste 
        e linhas que correspondem as features de treino

        k: k vizinhos mais próximos a se considerar para realizar a contagem

        col: a coluna que corresponde à alguma das features de teste

        """

        # Data
        aux = pd.DataFrame([df_distancias.iloc[:,col],df_distancias['labels']]).T

        # Retornar as previsões
        y_pred = int(aux.nsmallest(n = self.K, columns = col).labels.value_counts().index[0])

        return y_pred

    # Função que calcula as distâncias
    def distancias(self, X_test, X_train, y_train):
    
        matriz = np.zeros((self.X_train.shape[0],self.X_test.shape[0]));

        linha = 0; coluna = 0
        for test_point in self.X_test:

            linha = 0

            # Retorna a distância de cada test_point para inúmeros train_point e depois itera em test_point
            for train_point in self.X_train:

                matriz[linha, coluna] = self.distancia_euclidiana(test_point, train_point)

                # Iterar em linha
                linha += 1

            # Iterar em coluna 
            coluna += 1

        df_distancias = pd.DataFrame(matriz)

        df_distancias['labels'] = self.y_train

        return df_distancias
    
    # Função para calcular a distância euclidiana
    def distancia_euclidiana(self, att1, att2):
        # Inicialização da distância
        dist = 0

        # De acordo com o número de dimensões do espaço de características
        for i in range(len(att1)):

            # Distância total ao quadrado é igual à anterior ao quadrado mais a próxima ao quadrado
            dist += pow((att1[i] - att2[i]),2)

            # Distância Euclidiana é Norma L2, então tira-se a raiz quadrada
        return np.sqrt(dist)