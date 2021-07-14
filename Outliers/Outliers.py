from scipy import stats
pd.set_option('display.max_columns', None)

#Lista de features numericas para analisar outliers:
numft = ['price','yearOfRegistration','powerPS','kilometer','postalCode']

#Se puramente igualasse ele consideraria o mesmo objeto (foi o que deu aqui, sei lah)
data_zscore = data.copy()

# %% Criando dataframe que permite a retirada de outliers
for feat in numft:
    data_zscore[feat] = np.abs(stats.zscore(data_zscore[feat]))

#Definindo threshold de 3 (3 desvios padroes):
threshold = 3
#Isso aqui vai me retornar as colunas e linhas com outliers:
iout = np.where(data_zscore[numft] > 3)

#Se eu quiser visualizar nos dados originais algum outlier a partir disso:
#Para ver o original precisaria fazer o standardscaler reverso
data.iloc[iout[0][500],iout[1][500]]

#No iout ele pode retornar duplicidade de linhas caso em uma mesma linha ocorra mais de um outlier
#Portanto este pequeno codigo se utiliza do bultin set para retornar as linhas unicas
irowsunique = list(set(list(iout[0])))

#Limpeza a partir dos indeces de linhas e colunas retornados
data_nooutlier = data.drop(data.index[irowsunique])

#Visualizacao em boxplot para verificar se realmente deletou outliers
sns.boxplot(data_clean['yearOfRegistration'])

#Para visualizar se há valores faltantes em cada feature

np.sum(data_nooutlier.isnull())/data_nooutlier.shape[0]

#Para retornar apenas as linhas que não possuem valores faltantes

data_nomissing = data_nooutlier[data_nooutlier.isnull().any(axis = 1)==False]

# Não há mais valores faltantes

data_nomissing.isnull().sum().sum()

 # Quantas linhas retiramos em relação ao anterior

 data_nomissing.shape[0]/data_nooutlier.shape[0]