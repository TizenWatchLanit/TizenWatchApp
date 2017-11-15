import os
import numpy as np
from scipy import linalg
from numpy import matrix
import matplotlib.pyplot as plt

path = 'D:\\gearS2\\rate.csv'
doc = open(path, 'rb').read()

y = []
k = 1 + 16*108
while k < len(doc):
    y.append(doc[k])
    k = k + 16

q = 121
p = 120
Q = q - p # =
m = 200
M = m // 2
K = m - M + 1
N = len(y) - (q - M)
# [n+1,n+m]
count = 0
X = {} # trajectory matrix = base matrix
R = {} # e lag-covariance matrix
TM = {} # test matrix
D = {}
D_ = {}
u = {}
S = {}
for n in range(0, N - m + 1):
    X[n] = np.zeros((M, K), dtype=int)
    for i in range(n + 1, n + M + 1):
        count = i - 1
        for j in range(n + 1, n + K + 1):
            X[n][i - (n + 1)][j - (n + 1)] = y[count]
            count += 1

    R[n] = np.dot(X[n], X[n].T)
    U, s, V = linalg.svd(R[n])
    U, s, V = matrix(U), np.sqrt(s), matrix(V)
    '''Us, Vs, Xs = {}, {}, {}
    d = np.linalg.matrix_rank(X[n])
    for i in range(d):
        Us[i] = U[:, i]  # U
        Vs[i] = np.dot(X[n].T, (U[:, i] / s[i]))  # V
        Xs[i] = np.dot(s[i] * Us[i], (matrix(Vs[i]).T))  # Xi'''

    l = 9
    I1 = range(0, l)
    Us = matrix(U.T[ : 9]).T

    #newXs1 = np.zeros((m, K))
    #for i in I1:
    #    newXs1 = newXs1 + Xs[i]
    #print(newXs1.shape)


    TM[n] = np.zeros((M, Q), dtype=int)
    for i in range(n + p + 1, n + p + M + 1):
        count = i
        for j in range(n + p + 1, n + q + 1):
            TM[n][i - (n + p + 1)][j - (n + p + 1)] = y[count]
            count += 1
    D[n] = 0
    for j in range(p + 1, q + 1):
        D[n] += np.dot((TM[n].T[j - (p + 1)]).T, TM[n].T[j - (p + 1)]) - \
                np.dot(np.dot(np.dot((TM[n].T[j - (p + 1)]).T, Us), Us.T), TM[n].T[j - (p + 1)])
    D_[n] = D[n] / ( M * Q )
    n_ = n # e largest value of m <= n so
#that the hypothesis of no change is accepted
    D_Other = 0
    for j in range(0, K):
        D_Other += np.dot((X[n_].T[j]).T, X[n_].T[j]) - np.dot(np.dot(np.dot((X[n_].T[j]).T, Us), Us.T), X[n_].T[j])
    u[n] = D_Other / ( M * Q )

    S[n] = D_[n] / u[n]
    #print(D[n])

#print(TM[0])
#print(D)
#print(X[0])
S = list(S.values())
S = np.array(S).tolist()
for i in range(len(S)):
    S[i] = S[i][0][0]
print(S)
x = [i for i in range(len(S))]
plt.subplot(2, 1, 1)
plt.plot(x, S)

D = list(D.values())
D = np.array(D).tolist()
for i in range(len(D)):
    D[i] = D[i][0][0]
print(D)
x = [i for i in range(len(D))]
plt.subplot(2, 1, 2)
plt.plot(x, D)
plt.show()
print("Done")
