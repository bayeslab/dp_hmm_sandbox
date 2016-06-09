#Bayesian inference for finite mixture model
#No hyperparameter model
import random,math

#Generate Data
Data=[]
for i in range(20):
    Data.append(random.gauss(0,1.5))
for i in range(10):
    Data.append(random.gauss(10,1))


#number of clusters
k=2

class Data_iteration(object):
    def __init__(self,Data,a,b):
        self.Data=Data
        self.a=a
        self.b=b
        #prior:prop1~Beta(1,1)
        self.prop1=random.betavariate(self.a,self.b)
        print(self.prop1)
        total = 0
        for i in range(len(self.Data)):
            total += self.Data[i]
        self.mean=total/len(self.Data)
        sd_total = 0
        for i in range(len(self.Data)):
            sd_total += pow(self.Data[i]-self.mean, 2)
        self.sd=sd_total/len(self.Data)
        self.latent=[]
        for i in range(len(self.Data)):
            if (random.uniform(0,1)<=self.prop1):
                self.latent.append(1)
            else:
                self.latent.append(2)
    def update_latent(self,Gau1,Gau2):
        for i in range(len(self.Data)):
            cond_prob_in_1=self.prop1*Gau1.sigma*math.exp(-Gau1.sigma*pow(self.Data[i]-Gau1.mu,2)/2)
            cond_prob_in_2=(1-self.prop1)*Gau2.sigma*math.exp(-Gau2.sigma*pow(self.Data[i]-Gau2.mu,2)/2)
            prob_in_1=cond_prob_in_1/(cond_prob_in_1+cond_prob_in_2)
            if (random.uniform(0,1)<=prob_in_1):
                self.latent[i]=1
            else:
                self.latent[i]=2
    def get_latent(self):
        return self.latent
    def update_proportion(self):
        n_1=0
        for i in range(len(self.Data)):
            if(self.latent[i]==1):
                n_1+=1
        n_2=len(self.Data)-n_1
        self.a+=n_1
        self.b+=n_2
        print(self.a)
        print(self.b)
        self.prop1=random.betavariate(self.a,self.b)
        
class Gau(object):
    def __init__(self,a,b,m,alpha,Data):
        self.a=a
        self.b=b
        self.m=m
        self.alpha=alpha
        self.Data=Data
        self.sigma=random.gammavariate(self.a/2,self.b/2)
        self.mu=random.gauss(self.m,1/self.alpha/self.sigma)
    def update_sigma(self,latent,group):
        n=0
        sum_square=0
        for i in range(len(latent)):
            if (latent[i]==group):
                n+=1
                sum_square+=pow(self.Data[i]-self.mu,2)
        self.a+=n
        self.b+=sum_square
        self.sigma=random.gammavariate(self.a/2,self.b/2)
        print('sigma: ',self.sigma)
    def update_mu(self,latent,group):
        n=0
        sum_x=0
        for i in range(len(latent)):
            if (latent[i]==group):
                n+=1
                sum_x+=self.Data[i]
        self.m=(self.alpha*self.m+sum_x)/(self.alpha+n)
        self.alpha+=n
        self.mu=random.gauss(self.m,1/self.alpha/self.sigma)
        print('mu: ',self.mu)
        
        


#prior
#prop~Beta(1,1) special case for dirichilet
#sigma~Gamma(a/2,b/2)
#mu~N(m,1/alpha/sigma)


data=Data_iteration(Data,1,1)
Gau1=Gau(1,1,3,1,Data)
Gau2=Gau(1,1,10,2,Data)
for i in range(30):
    print('update',i)
    data.update_latent(Gau1,Gau2)
    data.update_proportion()
    Gau1.update_sigma(data.get_latent(),1)
    Gau2.update_sigma(data.get_latent(),2)
    Gau1.update_mu(data.get_latent(),1)
    Gau2.update_mu(data.get_latent(),2)



