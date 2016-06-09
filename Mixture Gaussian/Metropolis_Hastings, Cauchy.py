#Metropolis-Hastings Algorithm - Cauchy
#proposal distribution is N(x,b^2)

#Cauchy
#f(x)=1/pi*1/(1+x^2)

import math, random

#variable:
b=1
Data=[5]

#function
def Cauchy(k):
    return 1/math.pi/(1+k**2)

#Main
for i in range (1,1000):
    x=Data[i-1]
    print (x)
    y=random.gauss(x,b)
    r=min(1,Cauchy(y)/Cauchy(x))
    u=random.uniform(0,1)
    if (u <= r):
        Data.append(y)
    elif (u > r):
        Data.append(x)

print(Data)



