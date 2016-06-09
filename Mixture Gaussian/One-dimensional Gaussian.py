#One-dimensional Gaussian Mixture Model
#two Gaussian

import math
import random

Data=[]
for i in range(20):
    Data.append(random.gauss(0,2))
for i in range(10):
    Data.append(random.gauss(10,1))


iteration=20
prior_mu1=2
prior_sigma1=1
prior_mu2=15
prior_sigma2=3

class Gau(object):
    def __init__(self,mu,sigma,Data):
        self.mu=mu
        self.sigma=sigma
        self.Data=Data
    def probability(self,x):
        return math.exp(-pow(x-self.mu,2)/2/pow(self.sigma,2))/pow(2*math.pi*pow(self.sigma,2),0.5)
    def update(self,prob):
        prob_sum=0
        for i in range(len(prob)):
            prob_sum += prob[i]
        update_mu_sum=0
        update_sigma_sum=0
        for i in range(len(self.Data)):
            update_mu_sum += prob[i]*self.Data[i]
            update_sigma_sum += prob[i]*pow(self.Data[i]-self.mu,2)
        self.mu = update_mu_sum/prob_sum
        self.sigma = update_sigma_sum/prob_sum
    
Gau1 = Gau(prior_mu1,prior_sigma1,Data)
Gau2 = Gau(prior_mu2,prior_sigma2,Data)
for j in range(iteration):
    Process1=[]
    Process2=[]
    for i in range(len(Data)):
        prob_1=Gau1.probability(Data[i])
        prob_2=Gau2.probability(Data[i])
        sum_prob=prob_1+prob_2
        Process1.append(prob_1/sum_prob)
        Process2.append(prob_2/sum_prob)
    Gau1.update(Process1)
    Gau2.update(Process2)

total_prob_1=0
for i in range(len(Process1)):
    total_prob_1 += Process1[i]
pi_1=total_prob_1/len(Data)
pi_2=1-pi_1

print("Gaussian1: ",pi_1,Gau1.mu,Gau1.sigma)
print("Gaussian2: ",pi_2,Gau2.mu,Gau2.sigma)


