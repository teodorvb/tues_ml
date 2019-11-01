from pylab import *
from numpy import *

x = linspace(-1, 1, 500);
y = x**2 + random.normal(size=500)/4

a = sum((x - mean(x))*(y - mean(y)))/sum((x - mean(x))**2)
b = mean(y) - a*mean(x)

print("a is ",a)
print("b is ",b)

model_x = linspace(-1, 1, 2)
scatter(x, y)
plot(model_x, model_x *a + b, 'r-', linewidth=2)
show()
