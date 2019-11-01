from pylab import *
from numpy import *

# Commands to install thre required python packages on Ubintu
#
# sudo apt install python3 python3-pip
# pip(3) install numpy matplotlib pytk

dsize = 50000
x = linspace(0, 1, dsize)
y = x*2 + 1 + random.normal(size=dsize)/3

a = sum((x - mean(x))*(y - mean(y)))/sum((x - mean(x))**2)
b = mean(y) - a*mean(x)

print("a is ",a)
print("b is ",b)

model_x = linspace(0, 1, 2)
scatter(x, y)
plot(model_x, model_x *a + b, 'r-', linewidth=2)
show()

