rm(list=ls())
setwd('C:/GD_/USP/2ITAU_Trabalhando/Raciocinios/Ajuizamento/Cessao')

install.packages('tidyverse')
library(tidyverse)

ones = read.csv('ones.csv')
nums = read.csv('nums.csv')

# Subset com as variáveis ordinais ----------------------------------------
ordinallist = names(nums)

ordinaldf = nums

# Variáveis Binárias ----------------------------------------
bindata = ones #(Oswestry e SF não tem binária)

#N?o há relação de ordinalidade, logo, todos os dados do dataset bindata são
#Se N?o adicionarmos [], R N?o mantém a estrutura de fatores
bindata[] = lapply(bindata[], factor)

dfbind = cbind(ordinaldf,bindata)

# Coeficiente de correlação de ponto bisserial ----------------------------

library(polycor)

# Matriz de correlação  ---------------------------------------------------
rpb = matrix(NA, nrow = length(ordinaldf)*length(bindata), ncol = 1)
xaxis = matrix(NA, nrow = length(ordinaldf)*length(bindata), ncol = 1)
yaxis = matrix(NA, nrow = length(ordinaldf)*length(bindata), ncol = 1)
cont_Bin_0 = matrix(NA, nrow = length(ordinaldf)*length(bindata), ncol = 1)
cont_Bin_1 = matrix(NA, nrow = length(ordinaldf)*length(bindata), ncol = 1)
missing = matrix(NA, nrow = length(ordinaldf)*length(bindata), ncol = 1)
pvalue = matrix(NA, nrow = length(ordinaldf)*length(bindata), ncol = 1)

cont = 1
for (j in 1:length(bindata)){
  for (i in 1:length(ordinaldf)){
    
    var1 = ordinaldf[[i]]
    var2 = bindata[[j]]
    rpb[cont,] = polyserial(as.numeric(ordinaldf[[i]]), as.numeric(bindata[[j]]))
    pvalue[cont,] = cor.test(as.numeric(ordinaldf[[i]]),as.numeric(bindata[[j]]))$p.value
    xaxis[cont,] = names(ordinaldf[i])
    yaxis[cont,] = names(bindata[j])
    
    #Retornar contagem de valores zero e valores um da variável binária
    #Isto é importante para saber quantos dados temos na correlação
    #As vezes a correlação é grande só porque temos poucos dados
    aux = data.frame(bin = bindata[[j]], ord = ordinaldf[[i]])
    aux_ = aux[complete.cases(aux),]
    tbl = table(aux_[[1]]) #Contar apenas o número de zeros e ones do binário, que é a primeira coluna
    cont_Bin_0[cont,] = tbl[1] #Número de zeros
    cont_Bin_1[cont,] = tbl[2] #Número de ones
    
    #Ele retorna missing no boxplot se a pessoa respondeu a questão da ordinária, mas N?o respondeu a questão binária
    missingord = aux[!is.na(aux[[2]]),] #O ??ndice 2 se refere à variável ordinal
    missing[cont,] = sum(is.na(missingord[[1]])) #O ??ndice 1 se refere à variável binária, da qual deveremos contar os missing
    
    cont = cont + 1
  }
}

#Aqui finalmente é a matriz de correlação
pbmatrix = data.frame(cbind(yaxis, xaxis, rpb, pvalue, cont_Bin_0, cont_Bin_1, missing))
names(pbmatrix) = c("Binaria","Ordinal","rpb","pvalue","Bin_0","Bin_1","Missing")

write.csv(pbmatrix, 'outputdata/biseralcorrelation.csv', row.names = FALSE)

