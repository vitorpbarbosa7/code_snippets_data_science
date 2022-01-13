def one2(data):

    import pandas as pd 
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder

    one = OneHotEncoder(sparse = False)
    types = pd.DataFrame(data.dtypes, columns = ['type'])
    objvars = types[types['type']=='object'].index; objvars
    
    dados = one.fit_transform(data[objvars])
    
    # Tratamento dos nomes
    wnames = [f'x{num}' for num in range(0, data.shape[1])]
    tnames = list(data)
    di = dict(zip(wnames,tnames))
    dfwnames = pd.DataFrame(one.get_feature_names(), columns = ['wnames'])
    wnames_ = dfwnames['wnames'].replace(di, regex = True)
    
    nums = data.drop(columns = objvars, axis = 1)
    #(list(nums))
    
    ones = pd.DataFrame(data = dados, columns = wnames_)
    #(list(ones))
    
    final = pd.concat([nums, ones], axis = 1, ignore_index = True)
    
    all_names = list(nums) + list(ones)
    
    final.columns = all_names
    
    return final