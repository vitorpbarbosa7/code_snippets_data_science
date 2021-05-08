setwd('C:/GD/DS/1Pipeline/Geom_bar_CategoricalVariables')

# Não precisa atribuir a alguma variável, já vai automático 
load(file = 'Credit.RData')

# Executar o Script 01 antes de exectuar este

# Análise Exploratória de dados -------------------------------------------

library(ggplot2)

str(Credit)

# Podemos utilizar tableas de contingência para observar a relação entre variáveis 
# categóricas
table(Credit$CreditStatus,Credit$CheckingAcctStat)

# Podemos utilizar barras com geom_bar, position = 'dodge'
ggplot(Credit, aes(x = CreditStatus, (..count..) / sum(..count..))) + 
  geom_bar(aes(fill = CheckingAcctStat), position = 'dodge')

# Sem a porcentagem, ou seja, sem o (..count..) e sim com os valores absolutos
ggplot(Credit, aes(x = CreditStatus)) + 
  geom_bar(aes(fill = CheckingAcctStat), position = 'dodge')
