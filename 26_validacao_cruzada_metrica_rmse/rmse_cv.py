# Função para calcular o RMSE:
def rmse_cv(modelo, X_train, y_train, scoring = 'neg_mean_squared_error', cv = 5):
    
    """
    Função que retorna valores de rmse para 5 diferentes amostras na validação cruzada

    Parâmetros
    ----------------
    modelo: Modelo de aprendizado de máquina

	... fazer depois
    """
    
    rmse = np.sqrt(-cross_val_score(modelo, 
                                   X_train, 
                                   y_train,
                                   scoring = scoring,
                                   cv = cv))
    return(rmse)