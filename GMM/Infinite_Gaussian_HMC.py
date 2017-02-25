import random
import math
import statistics
import scipy.stats
import numpy as np
from scipy.stats import norm
import scipy.special
import matplotlib.pyplot as plt

#Function
def gamma(a,b):
    return random.gammavariate(a/2.,2.*b/a)

def gen_prob(count, parameter):
    prob={}
    denominator=sum(count.values())+parameter
    for key in count:
        prob[key]=count[key]/denominator
    prob['new']=parameter/denominator
    return prob

def normal_prob(prob):
    sum_prob=sum(prob.values())
    for key in prob:
        prob[key]=prob[key]/sum_prob
    return prob
        
def gen_state(prob):
    u=random.uniform(0,1)
    a=0
    for key in prob:
        a += prob[key]
        if (u <= a):
            return key

def beta_density_ratio(new_beta, beta, k, omega, gaussian):
    log_beta_product = 0
    for st in gaussian:
        log_beta_product += 0.5*(new_beta-beta)*(math.log(gaussian[st]['s']*omega)-gaussian[st]['s']*omega)
    log_ratio = -k*(math.lgamma(0.5*new_beta)-math.lgamma(0.5*beta))-0.5*(1/new_beta - 1/beta)+0.5*((k*new_beta-3)*math.log(0.5*new_beta)-(k*beta-3)*math.log(0.5*beta))+log_beta_product
    if (log_ratio > -1000 and log_ratio < 1):
        ratio = math.exp(log_ratio)
    elif (log_ratio >= 1):
        ratio = 1
    else:
        ratio = 0
    return ratio

def gaussian_pdf(x,mu,s):
    sd=pow(1./s,0.5)
    return scipy.stats.norm(mu,sd).pdf(x)

def alpha_Hamiltonian(alpha, p, k, n):
    log_density = (k-1.5)*math.log(alpha)-0.5/alpha+math.lgamma(alpha)-math.lgamma(alpha+n)
    return -log_density+pow(p,2.0)/2.0

def alpha_diff_U(alpha, k, n):
    return -(k-1.5)/alpha-0.5/pow(alpha,2.)-scipy.special.psi(alpha)+scipy.special.psi(alpha+n)

def graph(Data,gaussian):
    plt.gcf().clear()
    x=np.linspace(0,50,500)
    Data_x=Data
    Data_y=[0 for i in range(len(Data))]
    for i in gaussian:
        y=norm.pdf(x,loc=gaussian[i]['mu'],scale=pow(1./gaussian[i]['s'],0.5))
        plt.plot(x,y)
    plt.plot(Data_x,Data_y,'ro')
    plt.pause(0.5)
    


class Infinite_Gaussian(object):
    def __init__(self,Data):
        self.Data=Data
        self.n=len(Data)
        self.mean=statistics.mean(Data)
        self.sd=statistics.stdev(Data)

    def prior_alpha(self):
        alpha_inverse=gamma(1,1)
        self.alpha=1/alpha_inverse
        self.reject_alpha=0

    def prior_state(self):
        self.index=0
        self.count={}
        self.state=[]
        self.state.append(self.index)
        self.count[self.index]=1
        while (len(self.state) < self.n):
            outcome = gen_state(gen_prob(self.count,self.alpha))
            if (outcome == 'new'):
                self.index += 1
                self.count[self.index]=1
                self.state.append(self.index)
            else:
                self.count[outcome]+=1
                self.state.append(outcome)

    def prior_hyperparameter(self):
        self.lambd=random.gauss(self.mean,self.sd)
        self.ori_lambd=self.lambd
        self.r=gamma(1,pow(self.sd,-2.))
        self.ori_r=self.r
        self.beta=1/gamma(1,1)
        self.ori_beta=self.beta
        self.reject_beta=0
        self.omega=gamma(1,pow(self.sd,2.))
        self.ori_omega=self.omega

    def prior_gaussian(self):
        self.gaussian={}
        for st in self.state:
            if (self.gaussian.get(st,None)==None):
                mu=random.gauss(self.lambd, pow(1/self.r,0.5))
                s=gamma(self.beta,1/self.omega)
                self.gaussian[st]={'mu':mu,'s':s}

    def update_alpha(self):
        self.k=len(self.gaussian.keys())
        L=1000
        epsilon=0.01
        p_t = random.gauss(0,1)
        q = self.alpha
        p = p_t - epsilon * alpha_diff_U(q, self.k, self.n)/2.0
        for i in range(L):
            q = q + epsilon * p
            if (q <= 0):
                q=-q
                p=-p
            if (i != L):
                p = p - epsilon * alpha_diff_U(q, self.k, self.n)
        p = p - epsilon * alpha_diff_U(q, self.k, self.n)/2.0
        exponent = alpha_Hamiltonian(self.alpha,p_t,self.k,self.n) - alpha_Hamiltonian(q,p,self.k,self.n)
        if (exponent > 1):
            r = 1
        else:
            r = min(1, math.exp(exponent))
        if (random.uniform(0,1) <= r):
            self.alpha = q
        else:
            self.reject_alpha += 1

    def update_state(self):
        state=[]
        for i in range(self.n):
            count={}
            for st in self.state:
                if(count.get(st,None)==None):
                    count[st]=1
                else:
                    count[st]+=1
            prob={}
            for count_st in count:
                if (count_st == self.state[i]):
                    prob[count_st]=(count[count_st]-1)/(self.n-1+self.alpha)*pow(self.gaussian[count_st]['s'],0.5)*math.exp(-self.gaussian[count_st]['s']*pow(self.Data[i]-self.gaussian[count_st]['mu'],2.)/2.)
                else:
                    prob[count_st]=count[count_st]/(self.n-1+self.alpha)*pow(self.gaussian[count_st]['s'],0.5)*math.exp(-self.gaussian[count_st]['s']*pow(self.Data[i]-self.gaussian[count_st]['mu'],2.)/2.)
            new_mu=random.gauss(self.lambd, pow(1/self.r,0.5))
            new_s=gamma(self.beta,1/self.omega)
            prob['new']=self.alpha/(self.n-1+self.alpha)*gaussian_pdf(self.Data[i],new_mu,new_s)
            outcome = gen_state(normal_prob(prob))
            if (outcome == 'new'):
                self.index += 1
                self.gaussian[self.index]={'mu':new_mu,'s':new_s}
                state.append(self.index)
            else:
                state.append(outcome)
        self.state=state

    def update_gaussian(self):
        self.count={}
        for i in range(self.n):
            if (self.count.get(self.state[i],None)==None):
                self.count[self.state[i]]={'n':1,'y_sum':self.Data[i]}
            else:
                self.count[self.state[i]]['n']+=1
                self.count[self.state[i]]['y_sum']+=self.Data[i]
        new_gaussian={}
        for st in self.count:
            new_gaussian[st]={}
            new_mean=(self.count[st]['y_sum']*self.gaussian[st]['s']+self.lambd*self.r)/(self.count[st]['n']*self.gaussian[st]['s']+self.r)
            new_var=1/(self.count[st]['n']*self.gaussian[st]['s']+self.r)
            new_gaussian[st]['mu']=random.gauss(new_mean,pow(new_var,0.5))
            y_sq=0
            for i in range(self.n):
                if (self.state[i]==st):
                    y_sq += pow(self.Data[i]-new_gaussian[st]['mu'],2.)
            new_a=self.beta+self.count[st]['n']
            new_b=(self.beta+self.count[st]['n'])/(self.omega*self.beta+y_sq)
            new_gaussian[st]['s']=gamma(new_a,new_b)
        self.gaussian=new_gaussian
        self.k=len(self.gaussian.keys())

    def update_lambd(self):
        sum_mu=0
        for st in self.gaussian:
            sum_mu += self.gaussian[st]['mu']
        new_mean=(self.mean*pow(self.sd,-2)+self.r*sum_mu)/(pow(self.sd,-2)+self.k*self.r)
        new_var=1/(pow(self.sd,-2)+self.k*self.r)
        self.lambd = random.gauss(new_mean, pow(new_var, 0.5))

    def update_r(self):
        sum_mu_lambd=0
        for st in self.gaussian:
            sum_mu_lambd += pow(self.gaussian[st]['mu']-self.lambd, 2.)
        new_a = self.k + 1
        new_b = (self.k + 1)/(pow(self.sd,2)+sum_mu_lambd)
        self.r = gamma(new_a, new_b)

    def update_omega(self):
        sum_s=0
        for st in self.gaussian:
            sum_s += self.gaussian[st]['s']
        new_a = self.k * self.beta + 1
        new_b = (self.k * self.beta + 1)/(pow(self.sd,-2)+self.beta*sum_s)
        self.omega = gamma(new_a, new_b)

    def update_beta(self):
        y = -1
        while (y < 0):
            y = random.gauss(self.beta, 0.5)
        r = min(1, beta_density_ratio(y,self.beta,self.k,self.omega,self.gaussian))
        if (random.uniform(0,1) <= r):
            self.beta=y
        else:
            self.reject_beta+=1
            
        
            

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


Data1=Infinite_Gaussian(Data)
Data1.prior_alpha()
Data1.prior_state()
Data1.prior_hyperparameter()
Data1.prior_gaussian()


iteration = 1000
num_graph=10
count={}
for m in range(iteration):
    if (m % (iteration/num_graph) == 0):
        print(m)
        print(Data1.alpha)
        graph(Data, Data1.gaussian)
    Data1.update_alpha()
    Data1.update_state()
    Data1.update_gaussian()
    if (m > iteration-1000):
        num = len(Data1.gaussian.keys())
        if (count.get(num, None)==None):
            count[num]=1
        else:
            count[num]+=1  
    Data1.update_lambd()
    Data1.update_r()
    Data1.update_omega()
    Data1.update_beta()

print("alpha rejection rate: ", Data1.reject_alpha/iteration)
print("beta rejection rate: ", Data1.reject_beta/iteration)
print(count)


        
