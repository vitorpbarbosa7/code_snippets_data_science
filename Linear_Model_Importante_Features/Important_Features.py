def absolute_coefficients(X, y, normalize = False): 
    
    """
    Função para saber quais variáveis são mais importantes
    
    Parâmetros
    --------------
    X: Design Matrix
    
    y: Variáveis resposta
    
    normalize: Se será realizado Standartization ou não sobre as variáveis preditoras
    
    Obs: Para saber se são mais importantes, deve-se utilizar normalize = True
    
    """

    from sklearn import linear_model
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    # Criação de um modelo 
    modelo = linear_model.LinearRegression(normalize= False, fit_intercept = True)

    # Treinamento do modelo
    if normalize:
        modelo.fit(scaler.fit_transform(X), y)
    else:
        modelo.fit(X,y)
    
    # Positivo ou negativo
    sinal = np.where(modelo.coef_ > 0, "positivo", "negativo")

    # Imprimir coeficientes com seus valores absolutos
    for coef, var, sinal in sorted(zip(map(abs, modelo.coef_), X.columns[:-1], sinal), reverse = True):
        print("%6.3f, %s, %s" % (coef,var, sinal))