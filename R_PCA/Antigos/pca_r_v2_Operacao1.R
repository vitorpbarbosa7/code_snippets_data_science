# setwd('C:\\Users\\vitor\\Google Drive\\!Pós\\!Disciplinas\\PQI5858\\EstudoDeCaso\\R_PCA')

setwd('C:\\Users\\Vitor Barbosa\\Google Drive\\!Pós\\!Disciplinas\\PQI5858\\EstudoDeCaso\\R_PCA')
dados = read.csv('Operacao1.csv')
dadosPCA = dados[,-c(1,2,3,21,22)]

# ggplot ------------------------------------------------------------------

# Seção fixa dos gráficos -------------------------------------------------
# Gráficos ----------------------------------------------------------------
library(tidyverse)
library(ggrepel)
# install.packages('ggpmisc')
library(ggpmisc)
# install.packages('ggthemes')
library(ggthemes)
#Pacote para stat_cor
# install.packages('ggpubr')
library(ggpubr)
# Seções fixas dos gráficos -----------------------------------------------
theme = theme_bw(base_size = 15) + 
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))
# Dados Completos ---------------------------------------------------------
dados_completo = read.csv('dados_completo.csv')
names(dados_completo)[1] = "Index"
lista_nomes_vazoes = names(dados_completo)[-c(1,2,3,21,22,23)]

library(reshape2)
meltdata_completo = melt(data = dados_completo, 
                id.vars = c("Index","Hora","Dia"),
                measure.vars = lista_nomes_vazoes)
names(meltdata_completo)[4:5] = c("Variável","Valor")

#Destacar partes do gráfico
rect1 = data.frame(xmin = 0, xmax = 368, ymin = -Inf, ymax=+Inf)
rect2 = data.frame(xmin = 700, xmax = 765, ymin =-Inf, ymax=+Inf)
plot_completo = ggplot(meltdata_completo, aes(x = Index, y=Valor))+
  geom_point(aes(color = Variável), size = 1) + 
  xlab("Tempo (h)") +
  ylab("Variáveis") +
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


library(plotly)
ggplotly()

# Plot das duas vazões principais (entrada e saída de anidrido bru --------
plot_in_out = ggplot(subset(meltdata_completo, Variável %in% c("FC0104","FC0619")), 
                              aes(x = Index, y=Valor))+
  geom_point(aes(color = Variável), size = 1) + 
  xlab("Tempo (h)") +
  ylab("Vazão mássica (kg/h)") +
  theme

tiff("plot_in_out.tiff", res = 300, width = 2000*1.2, height = 1600)
plot_in_out
dev.off()



# Segunda operação não é estacionária? ----------------------------------------
plot_estacionario_falso = ggplot(subset(meltdata_completo, Index %in% c(700:765)), 
                       aes(x = Index, y=Valor))+
  geom_point(aes(color = Variável), size = 1) + 
  xlab("Tempo (h)") +
  ylab("Variáveis") +
  theme + 
  geom_vline(xintercept = 731, colour = "grey20")

tiff("plot_estacionario_falso.tiff", res = 300, width = 2000*1.2, height = 1600)
plot_estacionario_falso
dev.off()



# MERGE DATA OPERATION 1 AND OPERATION 2 ----------------------------------
df_bind = dados_completo[c(1:368,700:765),]
write.csv(df_bind, 'df_bind.csv')
# mERGED OPERATIONS --------------------------------------------------------------
lista_nomes_vazoes = names(df_bind)[-c(1,2,3,21,22,23)]

library(reshape2)
meltdata = melt(data = df_bind, 
                id.vars = c("Index","Hora","Dia"),
                measure.vars = lista_nomes_vazoes)
names(meltdata)[4:5] = c("Variável","Valor")

plot_geral = ggplot(meltdata, aes(y = Valor, x = Index)) + 
  geom_point(aes(color = Variável),na.rm = T)

plot_geral

  #Escrever o melt em csv:
write.csv(meltdata, 'meltdata.csv')


# Correlação --------------------------------------------------------------
# install.packages('corrplot')
# install.packages('Hmisc')
library(corrplot)
library(Hmisc)
#Correlação geral:

dadosPCA = df_bind[,-c(1,2,3,21,22,33)]
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

#Correlação com nomes dos conjuntos:
dados_grupos = read.csv('df_bind.csv', encoding = 'UTF-8')
dados_grupos = dados_grupos[,-c(1,2,3,21,22)]
corr = rcorr(as.matrix(dados_grupos))

tiff('corrclust_grupos.tiff', height = 3000, width = 3000, res = 300)
corrplot(corr$r, 
         method = 'square',
         order = 'hclust',
         addrect = 5,
         col=colorRampPalette(c("red","white","darkblue"))(200),
         tl.col = 'black')
dev.off()



corrplot(corr$r, 
         method = 'square',
         order = 'hclust',
         addrect = 5)


#Correlação por partes



# FACET FC0619 ------------------------------------------------------------

lista_melt_facet = names(df_bind)[-c(1,2,3,21,22,23,31)]

library(reshape2)
meltdata_facet = melt(data = dados_completo, 
                         id.vars = c("Index","Hora","Dia","FC0619"),
                         measure.vars = lista_nomes_vazoes)
names(meltdata_facet)[5:6] = c("Variável","Valor")

myplot = ggplot(meltdata_facet, aes(x = Valor, y=FC0619)) +
  geom_point(aes(color = Variável), size = 1) + 
  facet_wrap(.~Variável, scales = "free") + 
  ylab("Vazão de anidrido bruto") +
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
#PCA correlação
pcacorr=prcomp(df_bind[,-c(1,2,3,21,22,23)],scale=T)
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
#loadings para seleção das variáveis mais importantes
pca_load_corr = pcacorr$rotation
pca_load_corr

componente = 7
pca_load_oredered_1_corr = loadings[order(abs(loadings[,componente])),componente]
dotchart(pca_load_oredered_1_corr,
         cex = 0.7, xlab = "loadings", main = "loadings PC1")


biplot(pcacorr,scale = 0,cex=c(0.5,0.85))


#PCA covariância
pcavar = prcomp(dadosPCA, scale=F)