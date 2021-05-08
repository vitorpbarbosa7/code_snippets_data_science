def label_encoder(data):
    """Funcao que permite transformar todas as features do tipo object"""
    
    label_encoder = LabelEncoder()
    
    
    #Retornar, em formato de número, quais serao as features do tipo object que serao transformadas 
    obj_features = list(data.select_dtypes(include = 'object'))
    obj_ft = [data.columns.get_loc(x) for x in obj_features]
    
    #Pode ser necessario transformar para string, mas na maioria dos casos nao (afinal já é objeto)
#     def obj2str(df, features):
#     for a in features:
#         df[a] = df[a].astype(str)
#     return df

#     data = obj2str(data, object_features_train)
    
    #Funcao que realizara efetivamente a transformacao
    def le(df, obj_ft):
        le = LabelEncoder()
        for ft in obj_ft:
            df[:,ft] = le.fit_transform(df[:,ft])
        return df

    data_le = pd.DataFrame(le(data.values, obj_ft), columns = data.columns).astype(float)
    
    return data_le