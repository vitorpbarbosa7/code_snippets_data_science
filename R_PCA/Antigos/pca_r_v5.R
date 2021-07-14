setwd('C:\\Users\\vitor\\Google Drive\\!Pós\\!Disciplinas\\PQI5858\\EstudoDeCaso\\R_PCA')

# setwd('C:\\Users\\Vitor Barbosa\\Google Drive\\!Pós\\!Disciplinas\\PQI5858\\EstudoDeCaso\\R_PCA')
nomes = read.csv('nomesvariaveis.csv', encoding = 'UTF-8')
names(nomes)[1] = 'ID'

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
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"),
        legend.position = "none")
# Dados Completos ---------------------------------------------------------
dados_completo = read.csv('dados_completo.csv')
names(dados_completo)[1] = "Index"
lista_nomes_vazoes = names(dados_completo)[-c(1,2,3,21,22)]

library(reshape2)
meltdata_completo = melt(data = dados_completo, 
                id.vars = c("Index","Hora","Dia"),
                measure.vars = lista_nomes_vazoes)
names(meltdata_completo)[4:5] = c("Variável","Valor")

#Plot completo inicial sem os estacionários destacados
plot_completo = ggplot(meltdata_completo, aes(x = Index, y=Valor))+
  geom_point(aes(color = Variável), size = 1) + 
  xlab("Tempo (h)") +
  ylab("Variáveis") +
  theme

tiff("plot_completo.tiff", res = 300, width = 2000*1.2, height = 1600)
plot_completo
dev.off()

#Destacar partes do gráfico
xminimo_1 = 120
xmaximo_1 = 340
rect1 = data.frame(xmin = xminimo_1, xmax = xmaximo_1, ymin = -Inf, ymax=+Inf)
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
df_bind = dados_completo[c(xminimo_1:xmaximo_1,700:731),]
write.csv(df_bind, 'df_bind.csv')
df_bind = df_bind[,-c(21,22)]
df_bind$Index_bind = c(1:length(df_bind$Index))
# MERGED OPERATIONS --------------------------------------------------------------
lista_nomes_vazoes = names(df_bind)[-c(1,2,3,32)]

library(reshape2)
meltdata = melt(data = df_bind, 
                id.vars = c("Index","Hora","Dia","Index_bind"),
                measure.vars = lista_nomes_vazoes)
names(meltdata)[5:6] = c("Variavel","Valor")

# PROCV MELT DATA PARA ADICIONAR AS GRANDEZAS ----------------------------------------------------------
vlookuplist = read.csv('Lista_Nomes_PROCV.csv', encoding = 'UTF-8')
# meltdata_outliers = meltdata_outliers[,c(4,1,2,3,5)]
names(vlookuplist)[1] = 'ID'

meltdata_modified = merge(meltdata, vlookuplist, by.x = "Variavel", by.y = "ID", all.x = T, all.y = T)

plot_geral = ggplot(meltdata_modified, aes(y = Valor, x = Index_bind)) + 
  geom_point(aes(color = Variavel),na.rm = T) + 
  theme +
  xlab('Registros de medidas') + 
  ylab('Valores')

tiff("plot_pos_bind_sequencial.tiff", res = 300, width = 2000*1.2, height = 1600)
plot_geral
dev.off()

# RETIRAR OUTLIERS --------------------------------------------------------
#Nome do dataframe: meltdata_modified ou df_bind

#Plot das vazões  
ggplot(subset(meltdata_modified, Grandeza %in% "Vazao"), aes(x = Index, y = Valor)) + 
  geom_point(aes(color = Variavel))

#Boxplot de todos antes de retirar outliers:
boxplot_facet = ggplot(meltdata_modified, aes(x = Variavel, y = Valor)) + 
  geom_boxplot(aes(color = Grupo)) + 
  facet_wrap(.~Variavel, scales = "free") + 
  theme(strip.text.x = element_blank()) + 
  ylab('Valores') + 
  xlab(' ')

tiff("boxplot_facet_before_out.tiff", width = 4200/1.5, height = 3800/2, res = 300)
boxplot_facet
dev.off()

#Histograma de todos antes de retirar outliers:
histogram_facet = ggplot(meltdata_modified, aes(x = Valor)) + 
  geom_histogram(aes(y = ..density.., color = Grupo, fill = Grupo)) + 
  facet_wrap(.~Variavel, scales = "free") + 
  ylab('Frequência') + 
  xlab(' ')

tiff("histogram_facet_before_out.tiff", width = 4200/1.5, height = 3800/2, res = 225)
histogram_facet
dev.off()


#Retirar os outliers e plotar o boxplot novamente para comparar
nomesvariaveis = toString(unique(meltdata_modified$Variavel))
nomesvariaveis = unlist(strsplit(nomesvariaveis, ","))
df_bind_outliers = df_bind

for (variter in 4:ncol(df_bind_outliers)){
lower_bound_simples = mean(df_bind_outliers[,variter]) - 3*sd(df_bind_outliers[,variter])
upper_bound_simples = mean(df_bind_outliers[,variter]) + 3*sd(df_bind_outliers[,variter])

df_bind_outliers = df_bind_outliers[df_bind_outliers[,variter] > lower_bound_simples & df_bind_outliers[,variter] < upper_bound_simples,]
}

df_bind_outliers$Index_bind = NULL
df_bind_outliers$Index_bind = c(1:length(df_bind_outliers$Index))

#Melt do dataframe bind agoraa sem outliers:
lista_nomes_vazoes = names(df_bind_outliers)[-c(1,2,3,32)]
library(reshape2)
meltdata_outliers = melt(data = df_bind_outliers, 
                id.vars = c("Index","Hora","Dia","Index_bind"),
                measure.vars = lista_nomes_vazoes)
names(meltdata_outliers)[5:6] = c("Variavel","Valor")

#PROCV EM R PARA ADICIONAR OS GRUPOS:----------------------------------------------------------------
vlookuplist = read.csv('Lista_Nomes_PROCV.csv', encoding = 'UTF-8')
# meltdata_outliers = meltdata_outliers[,c(4,1,2,3,5)]
names(vlookuplist)[1] = 'ID'

m5 = merge(x = meltdata_outliers, y = vlookuplist, by.x = 'Variavel', by.y = 'ID', all.x = T, all.y = T)
#-------------------------------------------------------------------------------------------------------

#Histogramas e box plot após retirar outliers
boxplot_facet = ggplot(m5, aes(x = Variavel, y = Valor)) + 
  geom_boxplot(aes(color = Grupo)) + 
  facet_wrap(.~Variavel, scales = "free") + 
  theme(strip.text.x = element_blank()) + 
  ylab('Valores') + 
  xlab(' ')

tiff('boxplot_apos_remocao_outliers.tiff',  width = 4200/1.5, height = 3800/2, res = 300)
boxplot_facet
dev.off()

histogram_facet = ggplot(m5, aes(x = Valor)) + 
  geom_histogram(aes(y = ..density.., color = Grupo, fill = Grupo)) + 
  facet_wrap(.~Variavel, scales = "free") + 
  ylab("Frequência") + 
  xlab(' ')

tiff("histogram_apos_remocao_outliers.tiff", width = 4200/1.5, height = 3800/2, res = 225)
histogram_facet
dev.off()


# Cirandagem pesada, boxplots antes e depois juntos em um só gráfi --------

#Coluna para juntar os dois e diferenciar na hora do plot
meltdata_modified$L = "A"
m5$L = "B"

merged_boxplot = rbind(meltdata_modified, m5)

boxplot_facet = ggplot(merged_boxplot, aes(x = factor(L), y = Valor)) + 
  geom_boxplot(aes(color = Grupo)) + 
  facet_wrap(.~Variavel, scales = "free") + 
  theme(strip.text.x = element_blank()) + 
  ylab('Valores') +
  xlab(' ')

tiff('merged_boxplot.tiff',  width = 4200/1.5, height = 3800/2, res = 300)
boxplot_facet
dev.off()



# Plot de tudo após retirada dos OUTLIERS ---------------------------------
plot_no_outliers = ggplot(m5, aes(x = Index_bind, y=Valor))+
  geom_point(aes(color = Variavel), size = 1) + 
  xlab("Tempo (h)") +
  ylab("Variáveis") +
  theme

# Plot de variáveis com histogramas estranhos -----------------------------
plot_estranhas = ggplot(subset(m5, 
    Variavel %in% c("FC0104","FC0105","FC0108","TC0213","FC0501","TI0201","TI0203","TI0204")), 
    aes(x = Index_bind, y=Valor))+
  geom_point(aes(color = Variavel), size = 2) + 
  xlab("Registros de medidas") +
  ylab("Valores") +
  theme_bw()

tiff('plot_estranhas.tiff',  width = 4200/1.5, height = 3800/2, res = 300)
plot_estranhas
dev.off()


library(plotly)
ggplotly()

library(data.table)
DT = data.table(m5)
DT[,.N,by = Variavel]

#Plot teste de desvio padrão
library(tidyverse)
ggplot(df_bind, aes(x = Index, y = PI0102)) + 
  geom_point() + 
  geom_hline(yintercept = (mean(df_bind$PI0102))) + 
  geom_hline(yintercept = lower_bound_simples) + 
  geom_hline(yintercept = upper_bound_simples)

# Correlação --------------------------------------------------------------
# install.packages('corrplot')
# install.packages('Hmisc')
library(corrplot)
library(Hmisc)
#Correlação geral:

dadosPCA = meltdata_outliers[,-c(1,2,3,21,22)]
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