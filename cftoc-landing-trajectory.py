# -*- coding: utf-8 -*-

"""
Constrained Finite Time Optimal Control (CFTOC) Landing Trajectory

Finite-time QP optimization is used to generate a smooth and feasible landing trajectory for a gimble thrust rocket. 
The "mpc-falcon-controller.py" script will later use MPC to control a high-fidelity rocket model along this reference trajectory.
"""

import matplotlib.pyplot as plt
import numpy as np 
import pyomo.environ as pyo


"""

Program Initialization

"""

# Rocket constants
m = 27648
l = 70
J = 1 / 16 * m * l**2
g = 9.8
Fmax = 2 * 845 * 1000
dmax = 5 * np.pi / 180
tmin = -20 * np.pi / 180
tmax = 20 * np.pi / 180

# Define timestep and horizon
TS = 0.1
N = int(10 / TS)

# number of states and inputs
nz = 4
nu = 2

# state and input constraints
zMin = np.array([tmin, -100, -3000, -100])
zMax = np.array([tmax, 100, 0, 500])
uMin = np.array([0, -dmax])
uMax = np.array([Fmax, dmax])

# initial condition and terminal constraints
v0 = 205.2
alt0 = 1228
t0 = 10 * np.pi / 180
z0Bar = np.array([t0, 0, -alt0, v0])
zNBar = np.array([0, 0, 0, 0])



"""

Vertical Landing Trajectory for Non-Linear Rocket Model

"""

# Pyomo Environment
model = pyo.ConcreteModel()
model.tidx = pyo.Set(initialize=range(0, N+1)) # length of finite optimization problem
model.zidx = pyo.Set(initialize=range(0, nz))
model.uidx = pyo.Set(initialize=range(0, nu))

# Create state and input variables trajectory:
model.z = pyo.Var(model.zidx, model.tidx)
model.u = pyo.Var(model.uidx, model.tidx)

# Objective:

Q = np.array([[5,0,0,0],
              [0,5,0,0],
              [0,0,1,0],
              [0,0,0,1]])
model.cost = pyo.Objective(expr = sum(((model.z[i, t] ** 2) * Q[i, i]) + (model.u[1, t] ** 2) + ((model.u[0, t] / Fmax) ** 2) for i in model.zidx for t in model.tidx if t >= N), sense=pyo.minimize)

# Constraints:

model.constraint1 = pyo.Constraint(model.zidx, rule=lambda model, i: model.z[i, 0] == z0Bar[i])
model.constraint2 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[0, t+1] == model.z[0, t] + TS * model.z[1, t]
                                   if t < N else pyo.Constraint.Skip)
model.constraint3 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[1, t+1] == model.z[1, t] + TS * ((-(l/2) * model.u[0, t] * pyo.sin(model.u[1, t])) / J)
                                   if t < N else pyo.Constraint.Skip)
model.constraint4 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[2, t+1] == model.z[2, t] + TS * model.z[3, t]
                                   if t < N else pyo.Constraint.Skip)
model.constraint5 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[3, t+1] == model.z[3, t] + TS * (((m*g) - (model.u[0, t]*pyo.cos(model.u[1, t]))) / m)
                                   if t < N else pyo.Constraint.Skip)
model.constraint6 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[0, t] <= uMax[0]
                                   if t < N else pyo.Constraint.Skip)
model.constraint7 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[0, t] >= uMin[0]
                                   if t < N else pyo.Constraint.Skip)
model.constraint8 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[1, t] <= uMax[1]
                                   if t < N else pyo.Constraint.Skip)
model.constraint9 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[1, t] >= uMin[1]
                                   if t < N else pyo.Constraint.Skip)

model.constraint10 = pyo.Constraint(model.zidx, rule=lambda model, i:model.z[i, N] == zNBar[i])

model.constraint11 = pyo.Constraint(model.zidx, model.tidx, rule=lambda model, i, t: model.z[i, t] <= zMax[i]
                                   if t <= N else pyo.Constraint.Skip)

model.constraint12 = pyo.Constraint(model.zidx, model.tidx, rule=lambda model, i, t: model.z[i, t] >= zMin[i] 
                                   if t <= N else pyo.Constraint.Skip)                             

# Now we can solve:
results = pyo.SolverFactory('ipopt').solve(model).write()

# plot results
theta = [pyo.value(model.z[0,0])]
omega = [pyo.value(model.z[1,0])]
h = [pyo.value(model.z[2,0])]
v = [pyo.value(model.z[3,0])]
F = [pyo.value(model.u[0,0])]
delta = [pyo.value(model.u[1,0])]
d = np.array(delta) * (180/np.pi)
d = d.tolist()

for t in model.tidx:
    if t < N:
        theta.append(pyo.value(model.z[0,t+1]))
        omega.append(pyo.value(model.z[1,t+1]))
        h.append(pyo.value(model.z[2,t+1]))
        v.append(pyo.value(model.z[3,t+1]))
    if t < N-1:
        F.append(pyo.value(model.u[0,t+1]))
        d.append(pyo.value(model.u[1,t+1]))

plt.figure()
plt.subplot(2,1,1)
plt.plot(F)
plt.ylabel('Force')
plt.subplot(2,1,2)
plt.plot(d)
plt.ylabel('Delta (angle)')

plt.figure()
plt.subplot(4,1,1)
plt.plot(theta)
plt.ylabel('Theta')
plt.subplot(4,1,2)
plt.plot(omega)
plt.ylabel('Omega')
plt.subplot(4,1,3)
plt.plot(h)
plt.ylabel('h')
plt.subplot(4,1,4)
plt.plot(v)
plt.ylabel('v')
plt.show()

"""

Vertical Landing Trajectory for Linearized Rocket Model

"""

# constants
m = 27648
l = 70
J = 1 / 16 * m * l**2
g = 9.8
Fmax = 2 * 845 * 1000
dmax = 5 * np.pi / 180
tmin = -20 * np.pi / 180
tmax = 20 * np.pi / 180

# Define timestep and horizon
TS = 0.1
N = int(10 / TS)

# number of states and inputs
nz = 4
nu = 2

# state and input constraints
zMin = np.array([tmin, -100, -3000, -100])
zMax = np.array([tmax, 100, 0, 500])

# Modify delta range -> delta * force
dvalue = Fmax*dmax
uMin = np.array([0, -dvalue])
uMax = np.array([Fmax, dvalue])

# initial condition and terminal constraints
v0 = 205.2
alt0 = 1228
t0 = 10 * np.pi / 180
z0Bar = np.array([t0, 0, -alt0, v0])
zNBar = np.array([0, 0, 0, 0])

# Pyomo Environment
model = pyo.ConcreteModel()
model.tidx = pyo.Set(initialize=range(0, N+1)) # length of finite optimization problem
model.zidx = pyo.Set(initialize=range(0, nz))
model.uidx = pyo.Set(initialize=range(0, nu))

# Create state and input variables trajectory:
model.z = pyo.Var(model.zidx, model.tidx)
model.u = pyo.Var(model.uidx, model.tidx)

# Objective:
Q = np.array([[5,0,0,0],
              [0,5,0,0],
              [0,0,1,0],
              [0,0,0,1]])
model.cost = pyo.Objective(expr = sum(((model.z[i, t] ** 2) * Q[i, i]) + (model.u[1, t] ** 2) + ((model.u[0, t] / Fmax) ** 2) for i in model.zidx for t in model.tidx if t >= N), sense=pyo.minimize)

# Constraints (Linearized): 

model.constraint1 = pyo.Constraint(model.zidx, rule=lambda model, i: model.z[i, 0] == z0Bar[i])
model.constraint2 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[0, t+1] == model.z[0, t] + TS * model.z[1, t]
                                   if t < N else pyo.Constraint.Skip)
model.constraint3 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[1, t+1] == model.z[1, t] - TS * ((l/(2*J)) * model.u[1, t])
                                   if t < N else pyo.Constraint.Skip)
model.constraint4 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[2, t+1] == model.z[2, t] + TS * model.z[3, t]
                                   if t < N else pyo.Constraint.Skip)
model.constraint5 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[3, t+1] == model.z[3, t] + TS * (g - ((1/m) * model.u[0, t]))
                                   if t < N else pyo.Constraint.Skip)
model.constraint6 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[0, t] <= uMax[0]
                                   if t < N else pyo.Constraint.Skip)
model.constraint7 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[0, t] >= uMin[0]
                                   if t < N else pyo.Constraint.Skip)
model.constraint8 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[1, t] <= uMax[1]
                                   if t < N else pyo.Constraint.Skip)
model.constraint9 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[1, t] >= uMin[1]
                                   if t < N else pyo.Constraint.Skip)

model.constraint10 = pyo.Constraint(model.zidx, rule=lambda model, i:model.z[i, N] == zNBar[i])

model.constraint11 = pyo.Constraint(model.zidx, model.tidx, rule=lambda model, i, t: model.z[i, t] <= zMax[i]
                                   if t <= N else pyo.Constraint.Skip)

model.constraint12 = pyo.Constraint(model.zidx, model.tidx, rule=lambda model, i, t: model.z[i, t] >= zMin[i] 
                                   if t <= N else pyo.Constraint.Skip)                             

# Now we can solve:
results = pyo.SolverFactory('ipopt').solve(model).write()

# plot results
theta = [pyo.value(model.z[0,0])]
omega = [pyo.value(model.z[1,0])]
h = [pyo.value(model.z[2,0])]
v = [pyo.value(model.z[3,0])]
F = [pyo.value(model.u[0,0])]
delta = [pyo.value(model.u[1,0])]
d = np.array(delta) * (180/np.pi)
d = d.tolist()

for t in model.tidx:
    if t < N:
        theta.append(pyo.value(model.z[0,t+1]))
        omega.append(pyo.value(model.z[1,t+1]))
        h.append(pyo.value(model.z[2,t+1]))
        v.append(pyo.value(model.z[3,t+1]))
    if t < N-1:
        F.append(pyo.value(model.u[0,t+1]))
        d.append(pyo.value(model.u[1,t+1]))

plt.figure()
plt.subplot(2,1,1)
plt.plot(F)
plt.ylabel('Force')
plt.subplot(2,1,2)
plt.plot(d)
plt.ylabel('Delta (angle)')

plt.figure()
plt.subplot(4,1,1)
plt.plot(theta)
plt.ylabel('Theta')
plt.subplot(4,1,2)
plt.plot(omega)
plt.ylabel('Omega')
plt.subplot(4,1,3)
plt.plot(h)
plt.ylabel('h')
plt.subplot(4,1,4)
plt.plot(v)
plt.ylabel('v')
plt.show()
