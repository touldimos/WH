import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Properties of the fluid
K = 0.001                           #Pipe roughness [mm]
n = 1.31*10**-6                     #Viscosity [m2/s]
g = 9.81                            #Gravity [m2/s]
E = 784532117.25*100                #Pipe's elasticity [Pa]
Kf = 1.96*10**9                     #Fluid's elasticity [Pa]
ρ = 999.4                           #Fluid's density [kg/m3]
μ = 0.3                             #Pipe's freedom

#Initial conditions
elevd = 10
elevup = 30
Q = 1.988
D = 0.77
e = 0.01
l = 1000
hfl = 10                   #Percentage of local losses [%]

#Calculation conditions
nodes = 11
tclose = 0
hzero = 120
tmax = 50

Din = D - 2 * e 
A = (np.pi * Din ** 2) * 0.25
v = Q/A

def fcw(n, D, v, K):
    midf = []
    Re = v * D / n
    f = (1/(2*np.log10(K/(3.72*D*1000))))**2
    midf.append(f)
    for i in range(1, 10):
        midf.append((1/(2*np.log10((K/(3.72*D*1000)) + (2.51/(Re*np.sqrt(midf[i - 1]))))))**2)
    f = midf[-1]
    return f

f = fcw(n, Din, v, K)

# f0 = (1/(2*np.log10(K/(3.72*Din*1000))))**2

# def f(f):
#     Re = v*Din/1000/n
#     F = (1/(2*np.log10((K/(3.72*Din*1000)) + (2.51/(Re*np.sqrt(f))))))**2 - f
#     return F
# f = fsolve(f, f0)

# Re = 4 * np.abs(Q) / (np.pi * Din * n)
# rhs = -1.793 * np.log10((K / (3.7 * Din))**1.114 + 6.925/Re)
# f = (1 / rhs)**2

hf = f*l*v**2/(Din*2*g)
hl = hfl/100*hf
dh = hf + hl

a = np.sqrt(Kf/ρ)/(np.sqrt(1 + (Kf * D * (1 - μ**2))/(E * e)))
dx = l/(nodes - 1)
dt = dx/a
delel = (elevup - elevd)/(nodes - 1)
ca = (g * np.pi * D**2)/(4 * a)
step = np.arange(0, tmax + dt, dt)

x = np.zeros(nodes)
h = np.zeros(nodes)
hlow = np.zeros(nodes)
hhigh = np.zeros(nodes)
pipez = np.zeros(nodes)
head = np.zeros(nodes)

for i in range(nodes):
    x[i] = i * dx
    h[i] = hzero - dh * i / nodes
    hlow[i] = h[i]
    hhigh[i] = h[i]
    pipez[i] = elevup - delel * i
    head[i] = h[i] - pipez[i]
   
Qic = np.ones(nodes) * Q
Vic = np.ones(nodes) * v
Hic = h

def chars(nodes, Q, D, n, K, H, ca, dt, hzero, V0, t, tclose):
    def fcw(n, D, v, K):
        midf = []
        Re = v * D / n
        f = (1/(2*np.log10(K/(3.72*D*1000))))**2
        midf.append(f)
        for i in range(1, 10):
            midf.append((1/(2*np.log10((K/(3.72*D*1000)) + (2.51/(Re*np.sqrt(midf[i - 1]))))))**2)
        f = midf[-1]
        return f

    cm_ = np.zeros(nodes)
    cn_ = np.zeros(nodes)
    Hnew = np.zeros(nodes)
    Qnew = np.zeros(nodes)
    Vnew = np.zeros(nodes)
    for i in range(1, nodes):
        if Q[i - 1] != 0:
            f = fcw(n, D, v, K)
        else:
            f = 0
        cfa = f * dt / (np.pi * D ** 3/2)
        cm = Q[i - 1] + ca * H[i - 1] - cfa * Q[i - 1] * abs(Q[i - 1])
        cm_[i] = cm
    for i in range(0, nodes - 1):
        if Q[i + 1] != 0:
            f = fcw(n, D, v, K)
        else:
            f = 0
        cfb = f * dt / (np.pi * D ** 3/2)
        cn = Q[i + 1] - ca * H[i + 1] - cfb * Q[i + 1] * abs(Q[i + 1])
        cn_[i] = cn
    Hnew = 0.5 * (cm_ - cn_) / ca
    Hnew[0] = hzero    
    Qnew = cn_ + ca * Hnew
    Vnew = 4 * Qnew / (np.pi * D ** 2)
    Hnew = 0.5 * (cm_ - cn_) / ca
    Hnew[0] = hzero
    if t > tclose:
        Vnew[-1] = 0
    else:
        Vnew[-1] = V0 - t / tclose * V0
    Qnew[-1] = Vnew[-1] * np.pi * D ** 2 / 4
    Hnew[-1] = (cm_[-1] - Qnew[-1]) / ca
    return Hnew, Qnew, Vnew

H = []
Q = []
V = []

H.append(Hic)
Q.append(Qic)
V.append(Vic)

H.append(chars(nodes, Qic, Din, n, K, Hic, ca, dt, hzero, v, dt, tclose)[0])
Q.append(chars(nodes, Qic, Din, n, K, Hic, ca, dt, hzero, v, dt, tclose)[1])
V.append(chars(nodes, Qic, Din, n, K, Hic, ca, dt, hzero, v, dt, tclose)[2])

for i in range(1, len(step)):
    H.append(chars(nodes, Q[i], Din, n, K, H[i], ca, dt, hzero, v, step[i], tclose)[0])
    Q.append(chars(nodes, Q[i], Din, n, K, H[i], ca, dt, hzero, v, step[i], tclose)[1])
    V.append(chars(nodes, Q[i], Din, n, K, H[i], ca, dt, hzero, v, step[i], tclose)[2])

Head = pd.DataFrame(H)
Discharge = pd.DataFrame(Q)
Velocity = pd.DataFrame(V)

plt.plot(Head[10], 'g', zorder = 10)
plt.hlines(0, -10, len(step), 'black', zorder = 9)
plt.vlines(0, -abs(Head[10].min()), Head[10].max(), 'black', zorder = 8)
plt.legend(['Transient Head'])
plt.title('Node {}'.format(10))
plt.xlabel('Time [sec]')
plt.ylabel('Head [m]');