def LASSO_regularization(modelo, X_train, y_train, normalize = True):

    """
    EXPLICAR

    """

    import numpy as np 
    import pandas as pd 
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression, Ridge, LassoCV
    from sklearn.model_selection import cross_val_score
    
    if normalize:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()

        ft_names = X_train.columns

        X_train = scaler.fit_transform(X_train)

        X_train = pd.DataFrame(X_train, columns = ft_names)

    def rmse_cv(modelo):
        rmse = np.sqrt(-cross_val_score(modelo, 
                                       X_train, 
                                       y_train, 
                                       scoring = "neg_mean_squared_error", 
                                       cv = 5))
        return(rmse)


    modelo = LassoCV(alphas = [1, 0.1, 0.001, 0.00050]).fit(X_train, y_train)

    # Coeficientes LASSO
    coef = pd.Series(modelo.coef_, index = X_train.columns)

    # Coeficientes LASSO mais relevantes e menos relevantes para o modelo
    imp_coef = pd.concat([coef.sort_values().head(10), coef.sort_values().tail(10)]);

    matplotlib.rcParams['figure.figsize'] = (8,10)
    imp_coef.plot(kind = 'barh')
    plt.title('Coeficientes no modelo LASSO')
    
    return imp_coef