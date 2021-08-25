def padroniza_vars(data, intype = 'float64'):

	"""
	Função para padronizar determinado tipo de variáveis dentro do dataset

	Parâmetros:
	--------------------------
	data: Dataframe

	intype: tipo de dados (ex: 'float64', 'int', 'int64')

	"""
    intype_vars = data.dtypes[data.dtypes == intype].index

    medias = data[intype_vars].mean()

    stds = data[intype_vars].std()

    data[intype_vars] = (data[intype_vars] - medias)/stds

    return data