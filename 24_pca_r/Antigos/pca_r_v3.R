setwd('C:\\Users\\vitor\\Google Drive\\!P�s\\!Disciplinas\\PQI5858\\EstudoDeCaso\\R_PCA')

# setwd('C:\\Users\\Vitor Barbosa\\Google Drive\\!P�s\\!Disciplinas\\PQI5858\\EstudoDeCaso\\R_PCA')
nomes = read.csv('nomesvariaveis.csv', encoding = 'UTF-8')
names(nomes)[1] = 'ID'

# ggplot ------------------------------------------------------------------

# Se��o fixa dos gr�ficos -------------------------------------------------
# Gr�ficos ----------------------------------------------------------------
library(tidyverse)
library(ggrepel)
# install.packages('ggpmisc')
library(ggpmisc)
# install.packages('ggthemes')
library(ggthemes)
#Pacote para stat_cor
# install.packages('ggpubr')
library(ggpubr)
# Se��es fixas dos gr�ficos -----------------------------------------------
theme = theme_bw(base_size = 15) + 
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))
# Dados Completos ---------------------------------------------------------
dados_completo = read.csv('dados_completo.csv')
names(dados_completo)[1] = "Index"
lista_nomes_vazoes = names(dados_completo)[-c(1,2,3,21,22)]

library(reshape2)
meltdata_completo = melt(data = dados_completo, 
                id.vars = c("Index","Hora","Dia"),
                measure.vars = lista_nomes_vazoes)
names(meltdata_completo)[4:5] = c("Vari�vel","Valor")

#Plot completo inicial sem os estacion�rios destacados
plot_completo = ggplot(meltdata_completo, aes(x = Index, y=Valor))+
  geom_point(aes(color = Vari�vel), size = 1) + 
  xlab("Tempo (h)") +
  ylab("Vari�veis") +
  theme

tiff("plot_completo.tiff", res = 300, width = 2000*1.2, height = 1600)
plot_completo
dev.off()

#Destacar partes do gr�fico
rect1 = data.frame(xmin = 0, xmax = 368, ymin = -Inf, ymax=+Inf)
rect2 = data.frame(xmin = 700, xmax = 765, ymin =-Inf, ymax=+Inf)
plot_completo = ggplot(meltdata_completo, aes(x = Index, y=Valor))+
  geom_point(aes(color = Vari�vel), size = 1) + 
  xlab("Tempo (h)") +
  ylab("Vari�veis") +
  theme + 
  geom_rect(data=rect1, aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax),
            fill = "grey",
            alpha=0.5,
            inherit.aes = FALSE) + 
  geom_rect(data=rect2, aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax),
            fill = "grey",
            alpha=0.5,
            inherit.aes = FALSE)

tiff("plot_completo_destacado_2.tiff", res = 300, width = 2000*1.2, height = 1600)
plot_completo
dev.off()


nlibrary(plotly)
# ggplotly()

# Plot das duas vaz�es principais (entrada e sa�da de anidrido bru --------
plot_in_out = ggplot(subset(meltdata_completo, Vari�vel %in% c("FC0104","FC0619")), 
                              aes(x = Index, y=Valor))+
  geom_point(aes(color = Vari�vel), size = 1) + 
  xlab("Tempo (h)") +
  ylab("Vaz�o m�ssica (kg/h)") +
  theme

tiff("plot_in_out.tiff", res = 300, width = 2000*1.2, height = 1600)
plot_in_out
dev.off()



# Segunda opera��o n�o � estacion�ria? ----------------------------------------
plot_estacionario_falso = ggplot(subset(meltdata_completo, Index %in% c(700:765)), 
                       aes(x = Index, y=Valor))+
  geom_point(aes(color = Vari�vel), size = 1) + 
  xlab("Tempo (h)") +
  ylab("Vari�veis") +
  theme + 
  geom_vline(xintercept = 731, colour = "grey20")

tiff("plot_estacionario_falso.tiff", res = 300, width = 2000*1.2, height = 1600)
plot_estacionario_falso
dev.off()



# MERGE DATA OPERATION 1 AND OPERATION 2 ----------------------------------
df_bind = dados_completo[c(1:368,700:765),]
write.csv(df_bind, 'df_bind.csv')
# MERGED OPERATIONS --------------------------------------------------------------
lista_nomes_vazoes = names(df_bind)[-c(1,2,3,21,22)]

library(reshape2)
meltdata = melt(data = df_bind, 
                id.vars = c("Index","Hora","Dia"),
                measure.vars = lista_nomes_vazoes)
names(meltdata)[4:5] = c("Vari�vel","Valor")

plot_geral = ggplot(meltdata, aes(y = Valor, x = Index)) + 
  geom_point(aes(color = Vari�vel),na.rm = T)

plot_geral


# WRITE CSV MELT ----------------------------------------------------------

#Escrever o melt em csv:
write.csv(meltdata, 'meltdata.csv')

#Aprender PROCV nessa porra-------------------------------------------------------------------
meltdata_modified = read.csv('meltdata_modified.csv', encoding = 'UTF-8')
names(meltdata_modified)[4] = 'Variavel'
# RETIRAR OUTLIERS --------------------------------------------------------
#Nome do dataframe: meltdata_modified ou df_bind

#Plot das vaz�es
ggplot(subset(meltdata_modified, Grandeza %in% "Vazao"), aes(x = Index, y = Valor)) + 
  geom_point(aes(color = Variavel))

#Plot das press�es
ggplot(subset(meltdata_modified, Grandeza %in% "P"), aes(x = Index, y = Valor)) + 
  geom_point(aes(color = Variavel))

#Boxplot de todos:



#Media e 3 vezes desvio padr�o 

lower_bound = mean(df_bind[,4]) + sd(df_bind[,4])





# Correla��o --------------------------------------------------------------
# install.packages('corrplot')
# install.packages('Hmisc')
library(corrplot)
library(Hmisc)
#Correla��o geral:

dadosPCA = df_bind[,-c(1,2,3,21,22)]
corr = rcorr(as.matrix(dadosPCA))

tiff('corrclust.tiff', height = 3000, width = 3000, res = 300)
corrclust = corrplot(corr$r, 
         method = 'square',
         order = 'hclust',
         addrect = 5,
         col=colorRampPalette(c("red","white","darkblue"))(200),
         tl.col = 'black')
dev.off()

corrplot(corr$r, 
         method = 'number',
         order = 'hclust',
         addrect = 5,
         type = 'upper',
         col=colorRampPalette(c("red","white","darkblue"))(200),
         tl.col = 'black')
# FACET FC0619 ------------------------------------------------------------
 
lista_melt_facet = names(df_bind)[-c(1,2,3,21,22,31)]

library(reshape2)
meltdata_facet = melt(data = dados_completo, 
                         id.vars = c("Index","Hora","Dia","FC0619"),
                         measure.vars = lista_nomes_vazoes)
names(meltdata_facet)[5:6] = c("Vari�vel","Valor")

myplot = ggplot(meltdata_facet, aes(x = Valor, y=FC0619)) +
  geom_point(aes(color = Vari�vel), size = 1) + 
  facet_wrap(.~Vari�vel, scales = "free") + 
  ylab("Vaz�o de anidrido bruto") +
  xlab(" ") + 
  ylim(0,1200) + 
  geom_smooth(method = 'lm', se = T, formula = y ~ x) + 
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"),
        legend.position = "none")

tiff("facet_anidrido.tiff", width = 1600, height = 1200, res = 150)
myplot
dev.off()
# PCA ---------------------------------------------------------------------
#PCA correla��o
pcacorr=prcomp(df_bind[,-c(1,2,3,21,22)],scale=T)
summary(pcacorr)

names(pcacorr)
desvios_pca = pcacorr$sdev
loadings = pcacorr$rotation
medias = pcacorr$center

#Extraaindo mais valores do objeto PCA para realizar plots
library(factoextra)
pca_var2 = get_eigenvalue(pcacorr)
variancas = pca_var2$eigenvalue

# Screeplot ---------------------------------------------------------------
#Screeplot:
fviz_eig(pcacorr)
plot(pca_var2$cumulative.variance.percent,type = 'b')
abline(h=90,lty=2,lwd=1.5,col=2)



# Loadings Plot -----------------------------------------------------------
#loadings para sele��o das vari�veis mais importantes
pca_load_corr = pcacorr$rotation
pca_load_corr

componente = 7
pca_load_oredered_1_corr = loadings[order(abs(loadings[,componente])),componente]
dotchart(pca_load_oredered_1_corr,
         cex = 0.7, xlab = "loadings", main = "loadings PC1")


biplot(pcacorr,scale = 0,cex=c(0.5,0.85))


#PCA covari�ncia
pcavar = prcomp(dadosPCA, scale=F)