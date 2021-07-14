rm(list=ls())
setwd('C:/GD_/USP/2ITAU_Trabalhando/Raciocinios/Ajuizamento/Cessao')

library(tidyverse)
# install.packages('psych')
library(psych)

X = read.csv('ones.csv', encoding = 'UTF-8')
target = read.csv('label3.csv', encoding = 'UTF-8', sep = ';')

y = target$target

datacorr = cbind(X,y)


# Tabela de contingência --------------------------------------------------
library(gmodels)

freq22 = matrix(NA, nrow = length(datacorr)^2, ncol = 1)
#Matrix nome das variavei
denominador = matrix(NA, nrow = length(datacorr)^2, ncol = 1)
numerador = matrix(NA, nrow = length(datacorr)^2, ncol = 1)
qte_denom = matrix(NA, nrow = length(datacorr)^2, ncol = 2)

nums = matrix(NA, nrow = length(datacorr)^2, ncol = 4)
chi = matrix(NA, nrow = length(datacorr)^2, ncol = 1)

cont = 1
for (i in 1:length(datacorr)){
  for (j in 1: length(datacorr)){
    
    cross = CrossTable(datacorr[[j]],datacorr[[i]], format = "SAS", chisq = TRUE);
    
    freq22[cont,] = as.numeric(cross$prop.row[2,2])
    qte_denom[cont,c(1,2)] = c(as.numeric(cross$t[2,1]) + as.numeric(cross$t[2,2]),as.numeric(cross$t[1,1]) + as.numeric(cross$t[1,2]))
    
    denominador[cont,] = names(datacorr[j])
    numerador[cont,] = names(datacorr[i])
    
    nums[cont,c(1:4)] = c(as.numeric(cross$t[1,1]), as.numeric(cross$t[1,2]), as.numeric(cross$t[2,1]), as.numeric(cross$t[2,2]))
    
    chi[cont,] = as.numeric(cross$chisq[3])
    
    cont = cont + 1
  }
}

crosstable = data.frame(cbind(denominador, numerador, freq22, chi, qte_denom, nums))
#Para lidar com perda de variaveis de outras abas
df_crosstable = crosstable

crosstable = df_crosstable
names(crosstable) = c("denominador", "numerador", "Frequencia","Pvalor","Qtd_1_Denom", "Qtd_0_Demon",
                      "D0N0",'D0N1','D1N0','D1N1')
#Deletar as variaveis que podem confundir com nome de colunas do dataframe
#rm(denominador,numerador, freq22,chi,qte_denom,nums)

crosstable$Frequencia = as.numeric(crosstable$Frequencia)
crosstable$Pvalor = as.numeric(crosstable$Pvalor)
crosstable$Qtd_1_Denom = as.numeric(crosstable$Qtd_1_Denom)
crosstable$Qtd_0_Demon = as.numeric(crosstable$Qtd_0_Demon)
crosstable$D0N0 = as.numeric(crosstable$D0N0)
crosstable$D0N1 = as.numeric(crosstable$D0N1)
crosstable$D1N0 = as.numeric(crosstable$D1N0)
crosstable$D1N1 = as.numeric(crosstable$D1N1)

crosstable = crosstable[crosstable$Frequencia < 1,]

# Check de algum com a target y 
cross = CrossTable(datacorr$y ,datacorr$animal_gato, format = "SAS", chisq = TRUE);

cross$prop.row[2,2]


