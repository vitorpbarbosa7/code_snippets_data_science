

saddle = function(x,y){
-x*x + y*y
}

x = y = seq(-1, +1, length = 100)

z = outer(x,y,saddle)

persp(x,y,z)