import WH
from scipy.optimize import fsolve
import pandas as pd
from tabulate import tabulate
import numpy as np
import openpyxl
import os

#Properties of the fluid
k = 0.01                            #Pipe roughness [mm]
V = 1.31*10**-6                     #Viscosity [m2/s]
g = 9.81                            #Gravity [m2/s]
E = 784532117.25                    #Pipe's elasticity [Pa]
K = 1.96*10**9                      #Fluid's elasticity [Pa]
ρ = 999.4                           #Fluid's density [kg/m3]
μ = 0.3                             #Pipe's freedom
M = 10                              #Intervals #10
N = 1                               #Interval throught pipe
n = 3                               #Column Number [-]

local_losses = 10                   #Percentage of local losses [%]
V_cl = 20                           #Valve Closure [s]
PN = 120                            #Pressure nomitated [m]

wb = openpyxl.load_workbook('Data.xlsx')
data = wb['Data']
Data = []

for i in range(1, 8):
       Data.append(data.cell(row = i, column = n).value)
zus, p_extra, zds, l, D, e, Qus = Data
Din, A, v, f0 = WH.regcond(D, e, Qus, k)

def f(f):
    Re = v*Din/1000/V
    F = (1/(2*np.log10((k/(3.72*Din)) + (2.51/(Re*np.sqrt(f))))))**2 - f
    return F
f = fsolve(f, f0)
DH = WH.losses(f, l, v, Din, g, local_losses)

ground, Pz_us, Pz_ds, Pz =  WH.gr(zds, zus, l, M, N, DH, p_extra)
pz = Pz
st_pr = Pz_us - ground
dyn_pr = pz - ground
print("The pipe's downstream head is", round(Pz_ds[-1], 3), "m")

Δp, ΔP, t = WH.DP(K, ρ, D, μ, E, e, l, v, g, ground)
Pmax_j, Pmin_j = WH.jouk(dyn_pr, Δp)
Pmax_m, Pmin_m = WH.mich(l, v, g, V_cl, dyn_pr)
Pmax_Jouk, Pmin_Jouk, Pmax_Mich, Pmin_Mich = WH.Hammer(M, N, l, Pmax_j, Pmin_j, p_extra, Pmax_m, Pmin_m)

print(WH.print_table(Pmax_Jouk, Pmin_Jouk, Pmax_Mich, Pmin_Mich))
WH.show_fig(Pmax_Jouk, Pmin_Jouk, Pmax_Mich, Pmin_Mich, V_cl, l, M, PN)
WH.Check(Pmax_Jouk, PN)