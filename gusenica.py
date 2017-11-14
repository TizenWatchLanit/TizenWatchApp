import os
import numpy as np
from scipy import linalg
from numpy import matrix as m
from pandas import DataFrame as df
import matplotlib.pyplot as plt

path = 'D:\\gearS2\\rate.csv'
doc = open(path, 'rb').read()

y = []
k = 1 + 16*108
while k < len(doc):
    y.append(doc[k])
    k = k + 16
x = [i for i in range(len(y))]

L = 220
N = len(y)
K = N - L + 1
count = 0

X = np.zeros((L, K), dtype=int)
for i in range(L):
    count = i
    for j in range(K):        
        X[i][j] = y[count]
        count = count + 1

d = np.linalg.matrix_rank(X)
#U, s, V = np.linalg.svd(X, 0, 1);

S = np.dot(X, X.T)
U, s, V = linalg.svd(S)
U, s, V = m(U), np.sqrt(s), m(V)

'''x = [i for i in range(len(s))]
plt.plot(x, s)
plt.show()'''

Us, Vs, Xs = {}, {}, {}
for i in range(d):
    Us[i] = U[:,i] #U
    Vs[i] = np.dot(X.T, (U[:, i] / s[i])) #V
    Xs[i] = np.dot(s[i] * Us[i], (m(Vs[i]).T)) #Xi

'''newXs = {}
m = 10
p = int(len(Xs)/m)
for i in range(p):
    newXs[i] = np.zeros((L, K))
    for j in range(i*m, m*(i+1)):
        newXs[i] = newXs[i] + Xs[j];'''

m=9
I1 = range(0, m)
I2 = range(m, L)

newXs1 = np.zeros((L, K))
for i in I1:
    newXs1 = newXs1 + Xs[i]

newXs2 = np.zeros((L, K))
for i in I2:
    newXs2 = newXs2 + Xs[i]

sum_trend = []
for k in range(1, N+1):
    sum = 0
    if (1 <= k and k < L):
        for i in range(1, k+1):
            sum = sum + newXs1[i-1, k-i]
        sum_trend.append(sum/k)
    if (L <= k and k < K):
        for i in range(1, L+1):
            sum = sum + newXs1[i-1, k-1]
        sum_trend.append(sum/L)
    if (K <= k and k <= N):
        for i in range(1, N-k+2):
            sum = sum + newXs1[i+k-K-1, K-i]
        sum_trend.append(sum/(N-k+1))
print(N)
print(len(sum_trend))
print(sum_trend)

y = sum_trend;
x = [i for i in range(len(y))]

plt.plot(x, y)
plt.show()


#print(s[0] * Vs[0])
#print(np.dot(X.T, Us[0]))


'''X1 = np.zeros((L, K), dtype=int)
print(X)
for i in range(d):
    X1 = X1 + Xs[i]
print(X1)'''

