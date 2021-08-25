# Funções e utilitários para análise de dados do dataset German Credit Data. 

# Obs: Caso tenha problemas com a acentuação, consulte este link:
# https://support.rstudio.com/hc/en-us/articles/200532197-Character-Encoding

# Configurando o diretório de trabalho
# Coloque entre aspas o diretório de trabalho que você está usando no seu computador
# Não use diretórios com espaço no nome
# setwd("C:/FCD/BigDataRAzure/Cap15/Projeto")
# getwd()

# Função para converter variáveis numéricas para fator
quantize.num <- function(x, nlevs = 5, maxval = 1000, 
                         minval = 0, ordered = TRUE){
  cuts <- seq(min(x), max(x), length.out = nlevs + 1)
  cuts[1] <- minval
  cuts[nlevs + 1] <- maxval
  print(cuts)
  x <- cut(x, breaks = cuts, order_result = ordered)
}

# ?cut

# Nomeando as variáveis
colNames <- c("CheckingAcctStat",
              "Duration",
              "CreditHistory",
              "Purpose",
              "CreditAmount",
              "SavingsBonds",
              "Employment",
              "InstallmentRatePecnt",
              "SexAndStatus",
              "OtherDetorsGuarantors",
              "PresentResidenceTime",
              "Property",
              "Age",
              "OtherInstalments",
              "Housing",
              "ExistingCreditsAtBank",
              "Job",
              "NumberDependents",
              "Telephone",
              "ForeignWorker")

# Nomeando as colunas que serao transformadas
colNames2 <- append(colNames, c("Duration_f", "CreditAmount_f", "Age_f"))

# Função anônima para testar o tipo de dado de cada coluna no dataset
colTypes <- list(function(x) is.character(x), # CheckingAcctStat
                 function(x) is.numeric(x),   # Duration
                 function(x) is.character(x), # CreditHistory
                 function(x) is.character(x), # Purpose
                 function(x) is.numeric(x),   # CreditAmount
                 function(x) is.character(x), # SavingsBonds
                 function(x) is.character(x), # Employment
                 function(x) is.numeric(x),   # InstallmentRatePecnt
                 function(x) is.character(x), # SexAndStatus
                 function(x) is.character(x), # OtherDetorsGuarantors
                 function(x) is.numeric(x),   # PresentResidenceTime
                 function(x) is.character(x), # Property
                 function(x) is.numeric(x),   # Age
                 function(x) is.character(x), # OtherInstalments
                 function(x) is.character(x), # Housing
                 function(x) is.numeric(x),   # ExistingCreditsAtBank
                 function(x) is.character(x), # Job
                 function(x) is.numeric(x),   # NumberDependents
                 function(x) is.character(x), # Telephone
                 function(x) is.character(x), # ForeignWorker
                 function(x) is.numeric(x)    # CreditStatus
)


colTypes2 <- list(function(x) is.factor(x),  # CheckingAcctStat
                  function(x) is.numeric(x), # Duration
                  function(x) is.factor(x),  # CreditHistory
                  function(x) is.factor(x),  # Purpose
                  function(x) is.numeric(x), # CreditAmount
                  function(x) is.factor(x),  # SavingsBonds
                  function(x) is.factor(x),  # Employment
                  function(x) is.numeric(x), # InstallmentRatePecnt
                  function(x) is.factor(x),  # SexAndStatus
                  function(x) is.factor(x),  # OtherDetorsGuarantors
                  function(x) is.numeric(x), # PresentResidenceTime
                  function(x) is.factor(x),  # Property
                  function(x) is.numeric(x), # Age
                  function(x) is.factor(x),  # OtherInstalments
                  function(x) is.factor(x),  # Housing
                  function(x) is.numeric(x), # ExistingCreditsAtBank
                  function(x) is.factor(x),  # Job
                  function(x) is.numeric(x), # NumberDependents
                  function(x) is.factor(x),  # Telephone
                  function(x) is.factor(x),  # ForeignWorker
                  function(x) is.factor(x)   # CreditStatus
)

# Indicar se o fator esta ordenado
isOrdered  <- as.logical(c(T,
                           F,
                           T,
                           F,
                           F,
                           T,
                           T,
                           F,
                           F,
                           T,
                           F,
                           T,
                           F,
                           T,
                           T,
                           F,
                           T,
                           F,
                           T,
                           T))

# Ordem dos fatores
factOrder  <- list(list("A11", "A14", "A12", "A13"),
                   NA,
                   list("A34", "A33", "A30", "A32", "A31"),
                   NA,
                   NA,
                   list("A65", "A61", "A62", "A63", "A64"),
                   list("A71", "A72", "A73", "A74", "A75"),
                   NA,
                   NA,
                   list("A101", "A102", "A103"),
                   NA,
                   list("A124", "A123", "A122", "A121"),
                   NA,
                   list("A143", "A142", "A141"),
                   list("A153", "A151", "A152"),
                   NA,
                   list("A171", "A172", "A173", "A174"),
                   NA,
                   list("A191", "A192"),
                   list("A201", "A202"))


fact.set  <- function(inframe, metaframe){
  # Esta funcao transforma o dataset para garantir que os fatores estao definidos
  # e adiciona nomes as colunas.
  numcol <- ncol(inframe) - 1
  for(i in 1:numcol){
    if(!is.numeric(inframe[, i])){
      inframe[, i]  <- as.factor(inframe[, i])}
  }
  
  inframe[, 21] <- as.factor(inframe[, 21])
  colnames(inframe) <- c(as.character(metaframe[, 1]), "CreditStatus")
  inframe
}



equ.Frame <- function(in.frame, nrep){
  nrep <- nrep - 1
  # Cria o dataframe com numero igual de respostas positivas e negativas
  # e converte a coluna 21 para fator.  
  if(nrep > 0){
    posFrame  <- in.frame[in.frame[, "CreditStatus"] == 2, ]
    posFrame <- posFrame[rep(seq_len(nrow(posFrame)), nrep), ]
    in.frame <- rbind(in.frame, posFrame)
    #    in.frame <- data.frame(Map(function(x,y){rbind(x, rep(y, nrep))}, 
    #                               in.frame, posFrame))
  }
  in.frame
}

# Serialização
serList <- function(serlist){
  
  ## Mensagens em caso de erro 
  messages  <- c("Input nao eh uma lista ou tem comprimento maior que zero",
                 "Elementos de input da lista sao NULL ou comprimento maior que 1",
                 "A serializacao falhou")
  
  if(!is.list(serlist) | is.null(serlist) | length(serlist) < 1) {
    warning(messages[2])
    return(data.frame(as.integer(serialize(list(numElements = 0, payload = NA), connection = NULL))))}
  
  # Encontrando o numero de objetos
  nObj  <-  length(serlist)
  
  tryCatch(outframe <- data.frame(payload = as.integer(serialize(list(numElements = nObj, payload = serlist), connection=NULL))),
           error = function(e){warning(messages[3])
             outframe <- data.frame(payload = as.integer(serialize(list(numElements = 0, payload = NA), connection=NULL)))}
  )
  outframe
}

unserList <- function(inlist){

  # Mensagens em caso de erro 
  messages <- c("A coluna esta faltando ou nao eh do tipo correto",
                "Serializacao falhou",
                "Funcao encontrou um erro")
  
  # Checando o tipo de dado de entrada
  if(!is.integer(inlist$payload) | dim(inlist)[1] < 2 | 
     is.null(inlist$payload | inlist$numElements < 1)){
    warning(messages[1]) 
    return(NA)
  }
  
  
  tryCatch(outList <- unserialize(as.raw(inlist$payload)),
           error = function(e){warning(messages[2]); return(NA)})
  
  
  if(outList$numElements < 1 ) {warning(messages[3]); return(NA)}
  
  outList$payload
}
