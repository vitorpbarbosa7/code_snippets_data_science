setwd('C:\\\\Users\\vitor\\Google Drive\\!Pós\\!Disciplinas\\PQI5858\\EstudoDeCaso\\R_PCA')

dados = read.csv('dados.csv', sep=',')

dadosPCA = dados[,-c(1,19,20,32,33)]

#PCA correlação
pcacorr=prcomp(dadosPCA,scale=T)
summary(pcacorr)

names(pcacorr)
desvios_pca = pcacorr$sdev
loadings = pcacorr$rotation
medias = pcacorr$center

#Extraaindo mais valores do objeto PCA para realizar plots
library(factoextra)
pca_var2 = get_eigenvalue(pcacorr)
variancas = pca_var2$eigenvalue

#Screeplot:
fviz_eig(pcacorr)
plot(pca_var2$cumulative.variance.percent,type = 'b')
abline(h=90,lty=2,lwd=1.5,col=2)


#loadings para seleção das variáveis mais importantes
pca_load_corr = pcacorr$rotation
pca_load_corr


pca_load_oredered_1_corr = loadings[order(abs(loadings[,1])),1]
dotchart(pca_load_oredered_1_corr,
         cex = 0.7, xlab = "loadings", main = "loadings PC1")





#PCA covariância
pcavar = prcomp(dadosPCA, scale=F)