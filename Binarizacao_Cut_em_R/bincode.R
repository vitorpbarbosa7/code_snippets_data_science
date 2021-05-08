setwd('C:/GD/DS/1Formacao/1BigDataAnalytics_R_Azure_ML/15-Classificacao_R_Azure')

# German Credit Data - Avaliação de Risco de Crédito ----------------------

data = read.csv('credito.csv', stringsAsFactors = TRUE)

# Dados:
str(data)


# Feature Engineering para Binarização ------------------------------------

# Definição das variáveis para o bin
nlevs = 5
maxval = 1000 
minval = 0
ordered = TRUE

# Analisar a variável data$X6
summary(data$X6)
hist(data$X6)
x = data$X6

# O resultado dessa linha são 6 níveis, seis fatores, por exemplo no default
cuts <- seq(min(x), max(x), length.out = nlevs + 1)

# Retornou os valores cuts
print(cuts)

# Reajuste apenas dos valores mínimos e máximos?
# O valor mínimo é 0 por que? 
cuts[1] <- minval

# O último valor é o valor máximo
cuts[nlevs + 1] <- maxval

# Reajustado os valores mínimo e máximo
print(cuts)

x <- cut(x, 
         breaks = cuts, 
         order_result = ordered)

# Analisar o resultado do bin
data$X6 = x

# Agora temos fatores
str(data$X6)

# Distribuição?
# Não está bem equilibrado, os seja, os bins foram equidistantes e não propoprcionais
summary(data$X6)




