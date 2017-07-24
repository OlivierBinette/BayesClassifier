# Olivier Binette, July 23, 2017.

############################
# Main
############################
import numpy as np
from scipy.special import binom
from scipy.stats import beta

_binomial_coeffs = np.array([binom(n, j) for j in range(0,n+1)])

def f_value(u, v, coeffs):
    return np.dot(coeffs, \
                  [_binomial_coeffs[j]*_binomial_coeffs[k] * u**j * (1-u)**(n-j) * v**k * (1-v)**(n-k) \
                   for k in range(0,n+1) for j in range(0,n+1)])

def MCMC_42(n, data_x, data_l):
                
    def _N_0(j,k):
        s = 0
        for i in range(len(data_l)):
            s += (1-data_l[i]) * values[i][j+n*k]
        return s
    
    def _N_1(j,k):
        s = 0
        for i in range(len(data_l)):
            s += data_l[i] * values[i][j+n*k]
        return s
    
    def call(coeffs, i):
        return np.dot(values[i], coeffs)
    
    def basis_value(x, j, k):
        return _binomial_coeffs[j]*_binomial_coeffs[k] * x[0]**j * (1-x[0])**(n-j) * x[1]**k * (1-x[1])**(n-k)
        
    def log_likelihood(coeffs):
        e = np.dot(values, coeffs)
        return np.sum((1-data_l)*np.log(e) + data_l*np.log(1-e))

    values = np.array([[(basis_value(data_x[i], j,k)) for k in range(0,n+1) for j in range(0,n+1)] for i in range(len(data_x))])
        
    N_0 = np.array([_N_0(j,k) / (_N_0(j,k) + _N_1(j,k)) for k in range(0,n+1) for j in range(0, n+1)])
    N_1 = np.array([_N_1(j,k) / (_N_0(j,k) + _N_1(j,k)) for k in range(0,n+1) for j in range(0, n+1)])
    
    def run(n_iter, burn_in, r_step):
        alpha = 0.5
        kappa = 1

        actual = np.array([N_0[j+n*k] for k in range(0,n+1) for j in range(0,n+1)])
        prop = np.array([N_0[j+n*k] for k in range(0,n+1) for j in range(0,n+1)])

        accepted = 0.
        rejected = 0.
        summed = 0

        res = np.array([0. for k in range(0,n+1) for j in range(0,n+1)])
        memory = []

        for g in range(n_iter+burn_in):

            for to_change in range(0, (n+1)*(n+1)):
                prop[to_change] = beta.rvs(alpha + kappa * N_1[to_change], alpha + kappa * N_0[to_change])
                if np.log(np.random.rand()) < log_likelihood(prop) + beta.logpdf(actual[to_change],alpha + kappa * N_1[to_change], alpha + kappa * N_0[to_change]) \
                - log_likelihood(actual) - beta.logpdf(prop[to_change],alpha + kappa * N_1[to_change], alpha + kappa * N_0[to_change]):
                    actual[to_change] = prop[to_change]
                    accepted += 1
                else:
                    prop[to_change] = actual[to_change]
                    rejected += 1

            if g > burn_in and g % r_step == 0:
                print(g)
                memory.append(np.copy(actual))
                res += actual
                summed += 1

        res /= summed

        print("Acceptance ratio: ", accepted*100 / (accepted+rejected))
        return (memory, res)
    
    return run


############################
# Running the code
############################
n=8
N = 80
data_x = np.concatenate((np.stack((beta.rvs(15,10, size=N), beta.rvs(15,10, size=N)), axis=-1),np.stack((beta.rvs(10,15, size=N), beta.rvs(10,15, size=N)), axis=-1)))
data_l = np.array([0 for i in range(N)] + [1 for i in range(N)])

trace, mean = MCMC_42(n, data_x, data_l)(1000,200,20)


############################
# Plotting the results
############################
%matplotlib inline
import matplotlib.pyplot as plt

delta = 0.025
x = np.arange(0, 1, delta)
y = np.arange(0, 1, delta)
X, Y = np.meshgrid(x, y)

plt.figure()

for i in range(len(trace)):
    f = np.vectorize(lambda u,v: f_value(u,v, trace[i]))
    Z = f(X,Y)
    CS = plt.contour(X, Y, Z, [0.5], colors="black", alpha=0.1)
    
f = np.vectorize(lambda u,v: f_value(u,v, mean))
Z = f(X,Y)
CS = plt.contour(X, Y, Z, [0.5], colors="black")

plt.xticks([0,1])
plt.yticks([0,1])
plt.xlim([0,1])
plt.ylim([0,1])
plt.gca().set_aspect('equal', adjustable='box')

plt.scatter(np.transpose(data_x)[0][0:N], np.transpose(data_x)[1][0:N], color='blue', alpha=0.8)
plt.scatter(np.transpose(data_x)[0][N:], np.transpose(data_x)[1][N:], color='red', alpha=0.8)

plt.savefig('test.png')





