#Tratar os nomes para retirar os milimetros a direita
fun_sub = function(x){
  return(sub('-.*','',x))
}

names(df) = sapply(names(df), fun_sub)
