def VIF(X):

	""" Função que calcula e retorna o Variance Inflation Factor de um dataframe com atributos 

	Parâmetros
	--------------
	X : Dataset com as variáveis preditoras e sem a variável target
	"""

	# Para retornar o VIF
	from statsmodels.stats.outliers_influence import variance_inflation_factor
	# Para conseguir adicionar uma coluna de constante 1
	import statsmodels.api as sm

	# Adição da coluna constante
	X_constante = sm.add_constant(X)

	#Vetor com o valor de VIF para cada um dos vetores colunas de variáveis exógenas supostamente independentes do modelo
	vif = [variance_inflation_factor(X_constante.values, i) for i in range(X_constante.shape[1])]

	#vif[1:] porque a primeira coluna é em relação a intersecção
	vifdataframe = pd.DataFrame({'vif': vif[1:]}, index = X.columns).T

	return vifdataframe