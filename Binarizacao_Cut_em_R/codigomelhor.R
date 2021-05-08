# Quando dividimos por quartis
cutfunction = function(data,x){
  data[,x] = cut(data[,x], breaks = c(quantile(data[,x], probs = seq(0,1,by = 0.2))))
  return(data[,x])
}

numericas = unlist(lapply(data, is.numeric))
numericas = names(data[,numericas])
for (i in 1:length(numericas)){
  data[,numericas[i]] = cutfunction(data, numericas[i])
}
