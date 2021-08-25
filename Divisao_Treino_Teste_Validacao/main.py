def split(data, target_name, test_ratio = 0.7, valid_ratio = 0.1):

    """Função para dividir o dataset em partições de treino, validação e de teste"""
    
    X = data.drop(target_name, axis = 1)
    y = data[target_name]

    from sklearn.model_selection import train_test_split
    
    # Divisão entre treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_ratio)
    
    # Divisão entre treino e validação 
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = valid_ratio)
    
    return X, y, X_train, X_valid, X_test, y_train, y_valid, y_test


# Chamar a função:
X, y, X_train, X_valid, X_test, y_train, y_valid, y_test = split(data, target_name = target_name)