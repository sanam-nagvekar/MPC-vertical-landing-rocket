# -*- coding: utf-8 -*-

"""
Linearized Rocket Model Reachability Analysis
"""

import polytope as pt
import numpy as np
import matplotlib.pyplot as plt

'''

Helper Functions

'''

def minkowski_sum(X, Y):
    # Minkowski sum between two polytopes based on 
    # vertex enumeration. So, it's not fast for the
    # high dimensional polytopes with lots of vertices.
    V_sum = []
    if isinstance(X, pt.Polytope):
        V1 = pt.extreme(X)
    else:
        # assuming vertices are in (N x d) shape. N # of vertices, d dimension
        V1 = X
        
    if isinstance(Y, pt.Polytope):
        V2 = pt.extreme(Y)
    else:
        V2 = Y

    for i in range(V1.shape[0]):
        for j in range(V2.shape[0]):
            V_sum.append(V1[i,:] + V2[j,:])
    return pt.qhull(np.asarray(V_sum))

def precursor(Xset, A, Uset=pt.Polytope(), B=np.array([])):
    if not B.any():
        return pt.Polytope(Xset.A @ A, Xset.b)
        
    tmp  = minkowski_sum( Xset, pt.extreme(Uset) @ -B.T )
    return pt.Polytope(tmp.A @ A, tmp.b)
    
def Oinf(Xset, A, Wset=pt.Polytope()):
    Omega = Xset
    k = 0
    Omegap = precursor(Omega, A).intersect(Omega)
    while not Omegap == Omega:
        k += 1
        Omega = Omegap
        if not pt.is_empty(Wset):
            Omegap = pt.reduce(precursor(Omega, A, Wset=Wset).intersect(Omega))
        else:
            Omegap = pt.reduce(precursor(Omega, A).intersect(Omega))
    return Omegap

'''

Initialization

'''

from scipy.signal import place_poles

# Parameters
m = 27648
l = 70
J = (1/16)*m*l**2
g = 9.8
TS = 0.1
F_nom = m*g

# Constraints
x_u = np.array([20*np.pi/180, 100])
x_l = np.array([-20*np.pi/180, -100])
u_u = 5*np.pi/180
u_l = -5*np.pi/180

# Model matrices
A = np.array([[1, TS], [0, 1]])
B = np.array([[0], [-TS*(l/(2*J))*F_nom]])

# Constraint matrices
H_x = np.vstack((np.eye(2), -np.eye(2)))
h_x = np.concatenate((x_u, -x_l)).reshape((-1,1))
H_u = np.array([1, -1]).reshape((-1,1))
h_u = np.array([u_u, -u_l]).reshape((-1,1))

"""

Reachability Analysis

"""


""" Test Case 1 """

#constraints sets representaed as polyhedra
X = pt.Polytope(H_x,h_x)
U = pt.Polytope(H_u,h_u)

# Place Poles
p = (0.5,0.55)
K = place_poles(A,B,p).gain_matrix

# Check Eigenvalues
#print( np.linalg.eig(A - B @ k)[0])

A_cl = A - B@K
#P_2 = scipy.linalg.solve_discrete_lyapunov(A_cl, Q+K.T@R@K)
S = X.intersect(pt.Polytope(U.A@K, U.b))
O_inf = Oinf(S, A_cl)
Af = O_inf.A
bf = O_inf.b

# Plot 
fig, ax = plt.subplots()
pt.Polytope(Af,bf).plot(ax)
ax.autoscale_view()
ax.axis('equal')
plt.show()

# Print H-Rep.
print(O_inf)

""" Test Case 2 """

# Place Poles
p = (0.7,0.65)
K = place_poles(A,B,p).gain_matrix

# Check Eigenvalues
#print( np.linalg.eig(A - B @ k)[0])

A_cl = A - B@K
#P_2 = scipy.linalg.solve_discrete_lyapunov(A_cl, Q+K.T@R@K)
S = X.intersect(pt.Polytope(U.A@K, U.b))
O_inf2 = Oinf(S, A_cl)
Af2 = O_inf2.A
bf2 = O_inf2.b

# Plot 
fig, ax = plt.subplots()
pt.Polytope(Af2,bf2).plot(ax)
ax.autoscale_view()
ax.axis('equal')
plt.show()

# Print H-Rep.
print(O_inf2)


fig, ax = plt.subplots()
pt.Polytope(Af2,bf2).plot(ax)
pt.Polytope(Af,bf).plot(ax)
ax.legend(['O_inf2','O_inf1'])
ax.axis('equal')
plt.show()


""" Test Case 3 """

C = {}
# Since the Polytope package cannot handle non full-dimensional
# polytopes, we need to define an epsilon box around zero in
# order to run the code:
eps = 0.001
S = pt.box2poly([[-eps, eps], [-eps, eps]])

N = 10
fig, ax = plt.subplots()
PreS = precursor(S, A, U, B)
for j in range(N):
    C[j]= PreS.intersect(X)
    PreS = precursor(C[j], A, U, B)

X_0 = C[N-1]
C[N-1].plot(ax)
pt.Polytope(Af,bf).plot(ax)

ax.legend(['X_0', 'O_inf1'])
ax.axis('equal')
plt.show()

""" Test Case 4 """

def precursor_const_u(Xset, A, u, B=np.array([])):
    if not B.any():
        return pt.Polytope(Xset.A @ A, Xset.b)
        
    tmp  = minkowski_sum( Xset, -(B*u).reshape((1,-1)) )
    return pt.Polytope(tmp.A @ A, tmp.b)


C = {}
# Since the Polytope package cannot handle non full-dimensional
# polytopes, we need to define an epsilon box around zero in
# order to run the code:


P = 6
d = 5 * np.pi/180
fig, ax = plt.subplots()
PreS = precursor_const_u(O_inf, A, d, B)
for j in range(P):
  C[j] = PreS.intersect(X)
  PreS = precursor_const_u(C[j], A, d, B)

R_1 = C[P-1]
R_1 = R_1.intersect(X_0)
R_1.plot(ax)

pt.Polytope(Af,bf).plot(ax)
ax.legend(['R_1', 'O_inf1'])
ax.axis('equal')
plt.show()

""" Test Case 5 """

C = {}
# Since the Polytope package cannot handle non full-dimensional
# polytopes, we need to define an epsilon box around zero in
# order to run the code:


P = 3
d = 5 * np.pi/180
fig, ax = plt.subplots()
PreS = precursor_const_u(O_inf, A, d, B)
for j in range(P):
  C[j] = PreS.intersect(X)
  PreS = precursor_const_u(C[j], A, d, B)

R_2 = C[P-1]
R_2 = R_2.intersect(X_0)
R_1.plot(ax)
R_2.plot(ax)

pt.Polytope(Af,bf).plot(ax)
ax.legend(['R_1','R_2', 'O_inf1'])
ax.axis('equal')
plt.show()
