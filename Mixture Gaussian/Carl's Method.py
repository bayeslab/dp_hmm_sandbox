#One-dimensional Gaussian Mixture Model
#Carl's method

import math
import random

Data=[5,5,5,4,4,10,11,10.5,10,9.5]
k=2

class Data_Summary(object):
    def __init__(self,Data):
        self.Data=Data
    def mean(self):
        total = 0
        for i in range(len(Data)):
            total += Data[i]
        self.mean=total/len(Data)
        return self.mean
    def sd(self):
        sd_total = 0
        for i in range(len(Data)):
            sd_total += pow(Data[i]-self.mean, 2)
        self.sd=sd_total/len(Data)
        return self.sd
    def indicator(self,alpha,k):
        self.indicator=[1]
        counter1=0
        counter2=0
        for i in range(1,len(self.Data)):
            for j in range(len(self.indicator)):
                if (self.indicator[j]==1):
                    counter1+=1
                elif (self.indicator[j]==2):
                    counter2+=1
            prob_1=(counter1+alpha/k)/(len(self.Data)-1+alpha)
            prob_2=(counter2+alpha/k)/(len(self.Data)-1+alpha)
            is_1=prob_1/(prob_1+prob_2)
            if (random.uniform(0,1)<=is_1):
                self.indicator.append(1)
            else:
                self.indicator.append(2)
        return self.indicator
        

class Gau(object):
    def __init__(self,mu,sigma,Data):
        self.mu=mu
        self.sigma=sigma
        self.Data=Data
    def update_mu(self,lambd,r,indicator,group):
        n=0
        y_sum=0
        for i in range(len(indicator)):
            if (indicator[i]==group):
                n+=1
                y_sum+=self.Data[i]
        if(n==0):
            print("Delete the Class")
        elif(n!=0):
            y_avg=y_sum/n
            self.mu=random.gauss((y_avg*n*self.sigma+lambd*r)/(n*self.sigma+r),1/(n*self.sigma+r))
    def update_sigma(self,beta,omega,indicator,group):
        n=0
        y_sum_mu=0
        for i in range(len(indicator)):
            if (indiator[i]==group):
                n+=1
                y_sum_mu+=pow(self.Data[i]-self.mu,2)
        if(n==0):
            print("Delete the Class")
        elif(n!=0):
            self.sigma=random.gammavariate(beta+n,1/(1/(beta+n)*(omega*beta)+y_sum_mu))

def update_lambda(mu_1,mu_2,r,mu_y,sigma_y,k):
    new_mu=(mu_y*pow(sigma_y,-2)+r*(mu_1+mu_2))/(pow(sigma_y,-2)+k*r)
    new_sigma=1/(pow(sigma_y,-2)+k*r)
    return random.gauss(new_mu,new_sigma)

def update_r(mu_1,mu_2,lambd,mu_y,sigma_y,k):
    new_a=k+1
    new_b=1/((pow(sigma_y,2)+pow(mu_1-lambd,2)+pow(mu_2-lambd,2)/(k+1)))
    return random.gammavariate(new_a,new_b)

def update_omega(sigma_1,sigma_2,beta,sigma_y,k):
    new_a=k*beta+1
    new_b=1/((pow(sigma_y,-2)+beta*(sigma_1+sigma_2))/(k*beta+1))
    return random.gammavariate(new_a, new_b)

def update_



data = Data_Summary(Data)
mu_y=data.mean()
sigma_y=data.sd()
#lambda~N(mu_y,sigma_y)
prior_lambda=random.gauss(mu_y,sigma_y)
#r~Gamma(1,sigma_y^-2)
prior_r=random.gammavariate(1,pow(sigma_y,-2))
#mu~N(lambda,r^-1)
prior_mu_1=random.gauss(prior_lambda,1/prior_r)
prior_mu_2=random.gauss(prior_lambda,1/prior_r)
#beta^-1~Gamma(1,1)
prior_beta=1/random.gammavariate(1,1)
#omega~Gamma(1,sigma_y^2)
prior_omega=random.gammavariate(1,pow(sigma_y,2))
#sigma~Gamma(beta,w^-1)
prior_sigma1=random.gammavariate(prior_beta,1/prior_omega)
prior_sigma2=random.gammavariate(prior_beta,1/prior_omega)
#alpha^-1~Gamma(1,1)
prior_alpha=random.gammavariate(1,1)
#p(ci=j)=(n(-i,j)+a/k)/n-1+alpha
prior_indicator=data.indicator(prior_alpha,k)
print(prior_indicator)

