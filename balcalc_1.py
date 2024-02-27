#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
rho = 1.293
v = np.linspace(100, 1000, 50)
L = 0.0254
mu = 1.989e-5

Re = rho*L*mu*v

plt.plot(v, Re)
plt.hlines(1e-3, xmin=100, xmax=1000, color='r')
plt.hlines(1e-5, xmin=100, xmax=1000, color='r')

#%%
Cd = 0.28
A = np.pi*L**2/4
C = 0.5*Cd*rho*A

Fd = 0.5*Cd*rho*A*v
plt.plot(v, Fd)

#%%
theta0_deg = 1
theta0 = theta0_deg*np.pi/180
v0 = 840
vx0 = v0*np.cos(theta0)
vy0 = v0*np.sin(theta0)
x0 = 0
y0 = 1
m_gr = 35
m = m_gr*0.06479891
g = 9.82

start = 0
stop = 1
steps = 1000
dt = (stop-start)/steps
t = np.linspace(start, stop, steps)

#%%
"""
Vi har theta0, r0 och v0 givna.
Vi har newtons F=ma som ger oss a0.
Från a0 och v0 räknar vi ut v1.
Från a0
"""
#%%
vx, vy = np.zeros(len(t)), np.zeros(len(t))
vx[0], vy[0] = vx0, vy0

for i in range(len(t)):
    dvx = ax[i-1]*dt
    vx[i] = vx[i-1] + dvx


#%%
vx = vx0 / ( 1 + vx0*C*np.cos(theta)**2*t/m )
plt.plot(t, vx)

#%%

k1 = C*np.sin(theta)/m
k2 = g/np.sin(theta)

vy = -np.sqrt(k2/k1)*np.tan(np.arctan(-vy0*np.sqrt(k1/k2)) + np.sqrt(k1*k2)*t)

plt.plot(t, vy)

# %%
x, y = np.zeros(len(t)), np.zeros(len(t))
x[0], y[0] = x0, y0

for i in range(1, len(t)):
    dx = vx[i-1]*dt
    x[i] = x[i-1] + dx

    dy = vy[i-1]*dt
    y[i] = y[i-1] + dy


plt.plot(t, x)
plt.plot(t, y)
plt.figure()
# plt.plot(x,y)
# %%
above_ground = y>0
x = x[above_ground]
y = y[above_ground]

plt.plot(x,y)
# %%
