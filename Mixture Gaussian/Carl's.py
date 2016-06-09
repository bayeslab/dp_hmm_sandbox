#One-dimensional Gaussian Mixture Model
#Carl's method

import math
import random

#Generate Data
Data=[]
for i in range(20):
    Data.append(random.gauss(10,1))
for i in range(10):
    Data.append(random.gauss(0,1))

#function
def gamma(a,b):
    return random.gammavariate(a/2,2*b/a)


#number of clusters
k=2

class Data_sum(object):
    def __init__(self,Data,k):
        self.alpha=gamma(1,1)
        self.k=k
        self.Data=Data
        sum_total = 0
        var_total=0
        for i in range(len(self.Data)):
            sum_total += self.Data[i]
        self.mean=sum_total/len(self.Data)
        for i in range(len(self.Data)):
            var_total += pow(self.Data[i]-self.mean,2)
        self.sd=pow(var_total/len(self.Data),0.5)
        self.latent=[]
        for i in range(len(self.Data)):
            if (random.uniform(0,1)<=0.5):
                self.latent.append(1)
            else:
                self.latent.append(2)
    def update_alpha(self):
        y=0
        while(y<=0):
            y=random.gauss(self.alpha,1)
        p_y=pow(y,self.k-1.5)*math.exp(-0.5/y)*math.gamma(y)/math.gamma(len(self.Data)+y)
        p_alpha=pow(self.alpha,self.k-1.5)*math.exp(-0.5/self.alpha)*math.gamma(self.alpha)/math.gamma(len(self.Data)+self.alpha)
        r=min(1,p_y/p_alpha)
        u=random.uniform(0,1)
        if (u<=r):
            self.alpha=y
    def update_latent(self,Gau1,Gau2):
        n=len(self.Data)
        for i in range(n):
            n_1=0
            n_2=0
            if (self.latent[i]==1):
                for j in range(n):
                    if(self.latent[j]==1):
                        n_1+=1
                n_2=n-n_1
                cond_prob_1=pow(10,10)*(n_1-1+self.alpha/self.k)/(n+self.alpha-1)*pow(Gau1.sigma,0.5)*math.exp(-Gau1.sigma*pow(self.Data[i]-Gau1.mu,2)/2)
                cond_prob_2=pow(10,10)*(n_2+self.alpha/self.k)/(n+self.alpha-1)*pow(Gau2.sigma,0.5)*math.exp(-Gau2.sigma*pow(self.Data[i]-Gau2.mu,2)/2)
            elif(self.latent[i]==2):
                for j in range(n):
                    if(self.latent[j]==2):
                        n_2+=1
                n_1=n-n_2
                cond_prob_1=pow(10,6)*(n_1+self.alpha/self.k)/(n+self.alpha-1)*pow(Gau1.sigma,0.5)*math.exp(-Gau1.sigma*pow(self.Data[i]-Gau1.mu,2)/2)
                cond_prob_2=pow(10,6)*(n_2-1+self.alpha/self.k)/(n+self.alpha-1)*pow(Gau2.sigma,0.5)*math.exp(-Gau2.sigma*pow(self.Data[i]-Gau2.mu,2)/2)
            prob_1=cond_prob_1/(cond_prob_1+cond_prob_2)
            if (random.uniform(0,1)<=prob_1):
                self.latent[i]=1
            else:
                self.latent[i]=2

        
class Hyperparameter(object):
    def __init__(self,mu_y,sd_y,k):
        self.k=k
        self.mu_y=mu_y
        self.sd_y=sd_y
        self.lambd=random.gauss(self.mu_y,self.sd_y)
        self.r=gamma(1,pow(self.sd_y,-2))
        self.beta=1/gamma(1,1)
        self.omega=gamma(1,pow(self.sd_y,2))
    def update_lambd(self,Gau1,Gau2):
        new_mu=(self.mu_y*pow(self.sd_y,-2)+self.r*(Gau1.mu+Gau2.mu))/(pow(self.sd_y,-2)+self.k*self.r)
        new_sigma=1/(pow(self.sd_y,-2)+self.k*self.r)
        self.lambd=random.gauss(new_mu,new_sigma)
    def update_r(self,Gau1,Gau2):
        new_a=k+1
        new_b=(k+1)/(pow(self.sd_y,2)+pow(Gau1.mu-self.lambd,2)+pow(Gau2.mu-self.lambd,2))
        self.r=gamma(new_a,new_b)
    def update_omega(self,Gau1,Gau2):
        new_a=self.k*self.beta+1
        new_b=(self.k*self.beta+1)/(pow(self.sd_y,-2)+self.beta*(Gau1.sigma+Gau2.sigma))
        self.omega=gamma(new_a,new_b)
    def update_beta(self,Gau1,Gau2):
        y=0
        while (y<=0):
            y=random.gauss(self.beta,1)
        p_y=pow(math.gamma(y/2),-self.k)*math.exp(-0.5/y)*pow(y/2,(self.k*y-3)/2)*pow(Gau1.sigma*self.omega,y/2)*math.exp(-y*Gau1.sigma*self.omega/2)*pow(Gau2.sigma*self.omega,y/2)*math.exp(-y*Gau2.sigma*self.omega/2)
        p_beta=pow(math.gamma(self.beta/2),-self.k)*math.exp(-0.5/self.beta)*pow(self.beta/2,(self.k*y-3)/2)*pow(Gau1.sigma*self.omega,self.beta/2)*math.exp(-self.beta*Gau1.sigma*self.omega/2)*pow(Gau2.sigma*self.omega,self.beta/2)*math.exp(-self.beta*Gau2.sigma*self.omega/2)
        r=min(1,p_y/p_beta)
        u=random.uniform(0,1)
        if (u<=r):
            self.beta=y

        
class Gau(object):
    def __init__(self,Data,hyperparameter):
        self.Data=Data
        self.mu=random.gauss(hyperparameter.lambd,1/hyperparameter.r)
        self.sigma=gamma(hyperparameter.beta,1/hyperparameter.omega)
    def update_mu(self,latent,group,lambd,r):
        n=0
        y_sum=0
        for i in range(len(self.Data)):
            if (latent[i]==group):
                n+=1
                y_sum+=self.Data[i]
        y_avg=y_sum/n
        new_mean=(y_sum*self.sigma+lambd*r)/(n*self.sigma+r)
        new_sd=1/(n*self.sigma+r)
        self.mu=random.gauss(new_mean,new_sd)
    def update_sigma(self,latent,group,beta,omega):
        n=0
        y_square=0
        for i in range(len(self.Data)):
            if (latent[i]==group):
                n+=1
                y_square+=pow(self.Data[i]-self.mu,2)
        new_a=beta+n
        new_b=(beta+n)/(omega*beta+y_square)
        self.sigma=gamma(new_a,new_b)




##Main
data=Data_sum(Data,k)
hyperparameter=Hyperparameter(data.mean,data.sd,k)
Gau1=Gau(Data,hyperparameter)
Gau2=Gau(Data,hyperparameter)

##Update
for i in range(10):
    Gau1.update_mu(data.latent,1,hyperparameter.lambd,hyperparameter.r)
    Gau2.update_mu(data.latent,2,hyperparameter.lambd,hyperparameter.r)
    Gau1.update_sigma(data.latent,1,hyperparameter.beta,hyperparameter.omega)
    Gau2.update_sigma(data.latent,2,hyperparameter.beta,hyperparameter.omega)
    hyperparameter.update_lambd(Gau1,Gau2)
    hyperparameter.update_r(Gau1,Gau2)
    hyperparameter.update_omega(Gau1,Gau2)
    hyperparameter.update_beta(Gau1,Gau2)
    data.update_alpha()
    data.update_latent(Gau1,Gau2)
    
print(Gau1.mu,Gau1.sigma,Gau2.mu,Gau2.sigma)
    
