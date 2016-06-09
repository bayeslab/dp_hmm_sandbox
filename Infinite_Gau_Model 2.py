#One-dimensional Gaussian Mixture Model
#Carl's method

import math
import random
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

#Generate Data
Data=[]
for i in range(20):
    Data.append(random.gauss(20,1))
for i in range(10):
    Data.append(random.gauss(5,0.5))
for i in range(10):
    Data.append(random.gauss(15,1))
for i in range(10):
    Data.append(random.gauss(10,0.5))

#function
def gamma(a,b):
    return random.gammavariate(a/2.,2.*b/a)
def gau_prob(y,mu,sigma):
    return math.exp(-pow(y-mu,2)/2./pow(sigma,2))/pow(2.*math.pi*pow(sigma,2),0.5)
def graph(Data,gau_table):
    plt.gcf().clear()
    x=np.linspace(0,50,500)
    Data_x=Data
    Data_y=[0 for i in range(len(Data))]
    for i in range(len(gau_table)):
        y=norm.pdf(x,loc=gau_table[i][0],scale=pow(1./gau_table[i][1],0.5))
        plt.plot(x,y)
    plt.plot(Data_x,Data_y,'ro')
    plt.pause(3)
    

#prior number of clusters
k=5

class Data_sum(object):
    def __init__(self,Data,k):
        self.alpha=gamma(1,1)
        self.k=k
        self.Data=Data
        self.n=len(self.Data)
        sum_total = 0
        var_total=0
        for i in range(self.n):
            sum_total += self.Data[i]
        self.mean=sum_total/self.n
        for i in range(self.n):
            var_total += pow(self.Data[i]-self.mean,2)
        self.sd=pow(var_total/self.n,0.5)
        self.latent=[]
        for i in range(self.n):
            self.latent.append(random.randint(0,self.k-1))
    def update_alpha(self,k):
        self.k=k
        y=0
        while(y<=0):
            y=random.gauss(self.alpha,1)
        p_y=pow(y,self.k-1.5)*math.exp(-0.5/y)*math.gamma(y)/math.gamma(self.n+y)
        p_alpha=pow(self.alpha,self.k-1.5)*math.exp(-0.5/self.alpha)*math.gamma(self.alpha)/math.gamma(self.n+self.alpha)
        r=min(1,p_y/p_alpha)
        if (random.uniform(0,1)<=r):
            self.alpha=y
    def update_latent(self,Gaussian,hyperparameter):
        for i in range(len(self.latent)):
            temporary=[]
            for j in range(len(Gaussian.gau_mixture)):
                temporary.append([Gaussian.gau_mixture[j][2],0,0,0])
            for j in range(len(Gaussian.gau_mixture)):
                for k in range(len(self.latent)):
                    if (self.latent[k]==temporary[j][0]):
                        temporary[j][1]+=1
            for j in range(len(temporary)):
                if (self.latent[i]==temporary[j][0]):
                    temporary[j][2]=(temporary[j][1]-1)/(self.n-1+self.alpha)*pow(Gaussian.gau_mixture[j][1],0.5)*math.exp(-Gaussian.gau_mixture[j][1]*pow(self.Data[i]-Gaussian.gau_mixture[j][0],2)/2)
                elif (self.latent[i]!=temporary[j][0]):
                    temporary[j][2]=temporary[j][1]/(self.n-1+self.alpha)*pow(Gaussian.gau_mixture[j][1],0.5)*math.exp(-Gaussian.gau_mixture[j][1]*pow(self.Data[i]-Gaussian.gau_mixture[j][0],2)/2)
            I=0
            sum_mu=0
            sum_sigma=0
            for k in range(1000):
                mu=random.gauss(hyperparameter.ori_lambd,1/hyperparameter.ori_r)
                sigma=gamma(hyperparameter.ori_beta,1/hyperparameter.ori_omega)
                I+=gau_prob(self.Data[i],mu,1/sigma)
                sum_mu+=mu
                sum_sigma+=sigma
            new_mu=sum_mu/1000
            new_sigma=sum_sigma/1000
            new_prob=self.alpha/(self.n-1+self.alpha)*I/1000
            temporary.append([temporary[-1][0]+1,0,new_prob,0])
            sum_prob=0
            for j in range(len(temporary)):
                sum_prob+=temporary[j][2]
            for j in range(len(temporary)):
                temporary[j][3]=temporary[j][2]/sum_prob
            u=random.uniform(0,1)
            a=0
            for j in range(len(temporary)):
                a+=temporary[j][3]
                if (u<=a):
                    self.latent[i]=temporary[j][0]
                    if (self.latent[i]==temporary[-1][0]):
                        Gaussian.gau_mixture.append([new_mu,new_sigma,self.latent[i]])
                    break
                
            

class Hyperparameter(object):
    def __init__(self,mu_y,sd_y,k):
        self.k=k
        self.mu_y=mu_y
        self.sd_y=sd_y
        self.lambd=random.gauss(self.mu_y,self.sd_y)
        self.r=gamma(1,pow(self.sd_y,-2))
        self.beta=1/gamma(1,1)
        self.omega=gamma(1,pow(self.sd_y,2))
        self.ori_lambd=self.lambd
        self.ori_r=self.r
        self.ori_beta=self.beta
        self.ori_omega=self.omega
    def update_lambd(self,gau_table):
        sum_mu=0
        for i in range(len(gau_table)):
            sum_mu+=gau_table[i][0]
        new_mu=(self.mu_y*pow(self.sd_y,-2)+self.r*sum_mu)/(pow(self.sd_y,-2)+self.k*self.r)
        new_var=1/(pow(self.sd_y,-2)+self.k*self.r)
        self.lambd=random.gauss(new_mu,pow(new_var,0.5))
    def update_r(self,gau_table):
        mu_lambd=0
        for i in range(len(gau_table)):
            mu_lambd+=pow(gau_table[i][0]-self.lambd,2)
        new_a=k+1
        new_b=(k+1)/(pow(self.sd_y,2)+mu_lambd)
        self.r=gamma(new_a,new_b)
    def update_omega(self,gau_table):
        sum_sigma=0
        for i in range(len(gau_table)):
            sum_sigma+=gau_table[i][1]
        new_a=self.k*self.beta+1
        new_b=(self.k*self.beta+1)/(pow(self.sd_y,-2)+self.beta*sum_sigma)
        self.omega=gamma(new_a,new_b)
    def update_beta(self,gau_table):
        product_sigma=1
        for i in range(len(gau_table)):
            product_sigma*=pow(gau_table[i][1]*self.omega,self.beta/2)*math.exp(-self.beta*gau_table[i][1]*self.omega/2)
        log_beta=math.log(pow(math.gamma(self.beta/2),-self.k)*math.exp(-0.5/self.beta)*pow(self.beta/2,(self.beta*self.k-3)/2)*product_sigma)
        y=0
        while (y<=0):
            y=random.gauss(self.beta,1)
        product_sigma_y=1
        for i in range(len(gau_table)):
            product_sigma_y*=pow(gau_table[i][1]*self.omega,y/2)*math.exp(-y*gau_table[i][1]*self.omega/2)
        log_y=math.log(pow(math.gamma(y/2),-self.k)*math.exp(-0.5/y)*pow(y/2,(y*self.k-3)/2)*product_sigma_y)
        r=min(1,math.exp(log_y/log_beta))
        if (random.uniform(0,1)<=r):
            self.beta=y

class Gau(object):
    def __init__(self,Data,hyperparameter,k):
        self.Data=Data
        self.gau_mixture=[]
        for i in range(k):
            mu=random.gauss(hyperparameter.lambd,pow(1./hyperparameter.r,0.5))
            sigma=gamma(hyperparameter.beta,1/hyperparameter.omega)
            self.gau_mixture.append([mu,sigma,i])
    def update_mu(self,latent,lambd,r):
        Remove=[]
        for i in range(len(self.gau_mixture)):
            n=0
            y_sum=0
            for j in range(len(latent)):
                if(latent[j]==self.gau_mixture[i][2]):
                    n+=1
                    y_sum+=self.Data[j]
            if (n==0):
                Remove.append(i)
                continue
            y_avg=y_sum/n
            new_mean=(y_sum*self.gau_mixture[i][1]+lambd*r)/(n*self.gau_mixture[i][1]+r)
            new_var=1/(n*self.gau_mixture[i][1]+r)
            self.gau_mixture[i][0]=random.gauss(new_mean,pow(new_var,0.5))
        for i in range(len(Remove)-1,-1,-1):
            self.gau_mixture.remove(self.gau_mixture[Remove[i]])
    def update_sigma(self,latent,beta,omega):
        for i in range(len(self.gau_mixture)):
            n=0
            y_square=0
            for j in range(len(self.Data)):
                if(latent[j]==self.gau_mixture[i][2]):
                    n+=1
                    y_square+=pow(self.Data[j]-self.gau_mixture[i][0],2)
            new_a=beta+n
            new_b=(beta+n)/(omega*beta+y_square)
            self.gau_mixture[i][1]=gamma(new_a,new_b)


        
#Main
data=Data_sum(Data,k)
hyperparameter=Hyperparameter(data.mean,data.sd,k)
Gaussian=Gau(Data,hyperparameter,k)
graph(Data,Gaussian.gau_mixture)

#Update
for m in range(200):
    Gaussian.update_mu(data.latent,hyperparameter.lambd,hyperparameter.r)
    Gaussian.update_sigma(data.latent,hyperparameter.beta,hyperparameter.omega)
    hyperparameter.update_lambd(Gaussian.gau_mixture)
    hyperparameter.update_r(Gaussian.gau_mixture)
    hyperparameter.update_omega(Gaussian.gau_mixture)
    hyperparameter.update_beta(Gaussian.gau_mixture)
    data.update_alpha(len(Gaussian.gau_mixture))
    data.update_latent(Gaussian,hyperparameter)
    if(m%20==0):
        graph(Data,Gaussian.gau_mixture)

            
            
