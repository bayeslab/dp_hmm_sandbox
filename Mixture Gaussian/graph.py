import pylab
import numpy as np
from scipy.stats import norm

x = np.linspace(-10,10,1000)
y = norm.pdf(x, loc=2.5, scale=1.5)    # for example
pylab.plot(x,y)
pylab.show()

