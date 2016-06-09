import random
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

A_x=[]
A_y=[]
for i in range(10):
    A_x.append(random.gauss(2.5,1.5))
    A_y.append(0.01)


x = np.linspace(-10,10,1000)
plt.ion()
y = norm.pdf(x, loc=2.5, scale=1.5)    # for example
#pylab.plot(x,y)
plt.plot(A_x,A_y,'ro')
plt.plot(x,y)
plt.pause(2)
plt.gcf().clear()
y = norm.pdf(x, loc=0, scale=3)
#pylab.plot(x,y)
plt.plot(A_x,A_y,'ro')
plt.plot(x,y)
plt.pause(2)
