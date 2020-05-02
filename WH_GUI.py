from tkinter import *
from math import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pandas as pd

main = Tk()
main.title('Water Hammer')
main.iconbitmap(r'C:\Users\owner\Desktop\test\hammer.ico')

main.geometry('680x250')
Label(main,  text = "PN [atm]:", font=("Helvetica", 8)).grid(row=0, column=3)
Label(main, text = "Diameter(D) [mm]:", font=("Helvetica", 8)).grid(row=1, column=3)
Label(main, text = "Wall thickness (e) [mm]:", font=("Helvetica", 8)).grid(row=2, column=3)
Label(main, text = "Valve closure (τ) [s]:", font=("Helvetica", 8)).grid(row=3, column=3)
Label(main, text = "Local losses [%]:", font=("Helvetica", 8)).grid(row=4, column=3)
Label(main, text = "Elevation US [m]:", font=("Helvetica", 8)).grid(row=5, column=3)
Label(main, text = "Pressure US [m]:", font=("Helvetica", 8)).grid(row=6, column=3)
Label(main, text = "Elevation DS [m]:", font=("Helvetica", 8)).grid(row=7, column=3)
Label(main, text = "Pipe's Length (L) [m]:", font=("Helvetica", 8)).grid(row=8, column=3)
Label(main, text = "Discharge (Q) [m3/s]:", font=("Helvetica", 8)).grid(row=9, column=3)
Label(main, text = "Maximum pressure (P) [m]:", font=("Helvetica", 8)).grid(row=10, column=3)
Label(main, text = "Properties of the fluid", font=("Helvetica", 8)).grid(row=0)
Label(main, text = "Pipe's Roughness (k) [mm]", font=("Helvetica", 8)).grid(row=1)
Label(main, text = "Fluid's viscosity (V) [m2/s]", font=("Helvetica", 8)).grid(row=2)
Label(main, text = "Pipe's elasticity (E) [Pa]", font=("Helvetica", 8)).grid(row=3)
Label(main, text = "Fluid's elasticity (K) [Pa]", font=("Helvetica", 8)).grid(row=4)
Label(main, text = "Fluid's density (ρ) [kg/m3]", font=("Helvetica", 8)).grid(row=5)
Label(main, text = "Pipe's freedom (μ)", font=("Helvetica", 8)).grid(row=6)

num1 = Entry(main)
num1.insert(0, 120)
num2 = Entry(main)
num2.insert(0, 560)
num3 = Entry(main)
num3.insert(0, 41.2)
num4 = Entry(main)
num4.insert(0, 0)
num5 = Entry(main)
num5.insert(0, 10)
num6 = Entry(main)
num6.insert(0, 597.50)
num7 = Entry(main)
num7.insert(0, 66.32)
num8 = Entry(main)
num8.insert(0, 596.32)
num9 = Entry(main)
num9.insert(0, 357.89)
num10 = Entry(main)
num10.insert(0, 277.50)
num11 = Entry(main)
num11.insert(0, 0.01)
num12 = Entry(main)
num12.insert(0, 1.31*10**-6)
num13 = Entry(main)
num13.insert(0, 784532117.25)
num14 = Entry(main)
num14.insert(0, 1.96*10**9)
num15 = Entry(main)
num15.insert(0, 999.4)
num16 = Entry(main)
num16.insert(0, 0.3)
blank = Entry(main)

num1.grid(row=0, column=4)
num2.grid(row=1, column=4)
num3.grid(row=2, column=4)
num4.grid(row=3, column=4)
num5.grid(row=4, column=4)
num6.grid(row=5, column=4)
num7.grid(row=6, column=4)
num8.grid(row=7, column=4)
num9.grid(row=8, column=4)
num10.grid(row=9, column=4)
num11.grid(row=1, column=1)
num12.grid(row=2, column=1)
num13.grid(row=3, column=1)
num14.grid(row=4, column=1)
num15.grid(row=5, column=1)
num16.grid(row=6, column=1)
num16.grid(row=6, column=1)
blank.grid(row=10, column=4)

#Input
PN = str(num1.get())
D = str(num2.get())
e = str(num3.get())
V_cl = str(num4.get())
local_losses = str(num5.get())
zus = str(num6.get())
p_extra = str(num7.get())
zds = str(num8.get())
l = str(num9.get())
Qus = str(num10.get())
k = str(num11.get())
V = str(num12.get())
E = str(num13.get())
K = str(num14.get())
ρ = str(num15.get())
μ = str(num16.get())
g = 9.81
g = 9.81
M = 10
N = 1

def WH():
    PN = float(num1.get())
    D = float(num2.get())
    e = float(num3.get())
    V_cl = float(num4.get())
    local_losses = float(num5.get())
    zus = float(num6.get())
    p_extra = float(num7.get())
    zds = float(num8.get())
    l = float(num9.get())
    Qus = float(num10.get())
    k = float(num11.get())
    V = float(num12.get())
    E = float(num13.get())
    K = float(num14.get())
    ρ = float(num15.get())
    μ = float(num16.get())

    blank.delete(0, END)

    Pz_us = zus + p_extra 
    slope = (zds - zus)/l
    Din = D - 2*e
    z = np.zeros(M + 1)

    for i in range(0, M + 1, N):
        z[0] = zus
        z[i] = z[i-1] + l/M*slope
    ground = z

    A = (np.pi*(Din/1000)**2)*0.25
    v = Qus/1000/A
    Re = v*Din/1000/V
    f0 = (1/(2*np.log10(k/(3.72*Din))))**2

    def f1(f):
        F = (1/(2*np.log10((k/(3.72*Din)) + (2.51/(Re*np.sqrt(f))))))**2 - f
        return F
    f = fsolve(f1, f0)

    dh = f*l*v**2/(Din/1000*2*g)
    dh_loc = local_losses/100*dh
    DH = dh + dh_loc
    Pz_ds = Pz_us - DH
    j = DH/l

    a = np.sqrt(K/ρ)/(np.sqrt(1 + (K*(D/1000)*(1 - μ**2))/(E*e/1000)))
    t = 2*l/a
    Δp = a*v/g
    ΔP = ground + Δp
    Pz = np.zeros(M + 1)
    for i in range(0, M + 1, N):
        Pz[0] = Pz_us
        Pz[i] = Pz[i - 1] - j*l/M

    pz = Pz
    st_pr = Pz_us - ground
    dyn_pr = pz - ground

    lp = np.zeros(M + 1)
    for i in range(0, M + 1, N):
        lp[0] = 0
        lp[i] = l/M*i

    if V_cl < 4:
        Pmax_j = max(dyn_pr) + Δp
        Pmin_j = max(dyn_pr) - Δp
        PPmax_J = (Pmax_j - p_extra)/(lp[-1] - lp[0])
        PPmin_J = (Pmin_j - p_extra)/(lp[-1] - lp[0])
        Pmax_Jouk = p_extra + PPmax_J*lp
        Pmin_Jouk = p_extra + PPmin_J*lp
        P = Pmax_Jouk
    else:
        Pm = 2*l*v/(g*V_cl)
        P_m = Pm
        Pmax_m = max(dyn_pr) + P_m
        Pmin_m = max(dyn_pr) - P_m
        PPmax_M = (Pmax_m - p_extra)/(lp[-1] - lp[0])
        PPmin_M = (Pmin_m - p_extra)/(lp[-1] - lp[0])
        Pmax_Mich = p_extra + PPmax_M*lp
        Pmin_Mich = p_extra + PPmin_M*lp
        P = Pmax_Mich
    if V_cl < 4:
        Label(main, text = "Joukowski (Sudden Closure)", font=("Helvetica", 8)).grid(row=3, column=5)
    else:
        Label(main, text = "Michaud (Slow Closure)", font=("Helvetica", 8)).grid(row=3, column=5)

    if P[-1] < PN:
        Label(main, text = "Pipe is sufficient", font=("Helvetica", 8)).grid(row=10, column=5)
    else:
            Label(main, text = "Pipe is not sufficient", font=("Helvetica", 8)).grid(row=10, column=5)
    blank.insert(0, round(P[-1], 3))
    # return P

P = WH()

def plot():
    PN = float(num1.get())
    D = float(num2.get())
    e = float(num3.get())
    V_cl = float(num4.get())
    local_losses = float(num5.get())
    zus = float(num6.get())
    p_extra = float(num7.get())
    zds = float(num8.get())
    l = float(num9.get())
    Qus = float(num10.get())
    k = float(num11.get())
    V = float(num12.get())
    E = float(num13.get())
    K = float(num14.get())
    ρ = float(num15.get())
    μ = float(num16.get())

    blank.delete(0, END)

    Pz_us = zus + p_extra 
    slope = (zds - zus)/l
    Din = D - 2*e
    z = np.zeros(M + 1)

    for i in range(0, M + 1, N):
        z[0] = zus
        z[i] = z[i-1] + l/M*slope
    ground = z

    A = (np.pi*(Din/1000)**2)*0.25
    v = Qus/1000/A
    Re = v*Din/1000/V
    f0 = (1/(2*np.log10(k/(3.72*Din))))**2

    def f1(f):
        F = (1/(2*np.log10((k/(3.72*Din)) + (2.51/(Re*np.sqrt(f))))))**2 - f
        return F
    f = fsolve(f1, f0)

    dh = f*l*v**2/(Din/1000*2*g)
    dh_loc = local_losses/100*dh
    DH = dh + dh_loc
    Pz_ds = Pz_us - DH
    j = DH/l

    a = np.sqrt(K/ρ)/(np.sqrt(1 + (K*(D/1000)*(1 - μ**2))/(E*e/1000)))
    t = 2*l/a
    Δp = a*v/g
    ΔP = ground + Δp
    Pz = np.zeros(M + 1)
    for i in range(0, M + 1, N):
        Pz[0] = Pz_us
        Pz[i] = Pz[i - 1] - j*l/M

    pz = Pz
    st_pr = Pz_us - ground
    dyn_pr = pz - ground

    lp = np.zeros(M + 1)
    for i in range(0, M + 1, N):
        lp[0] = 0
        lp[i] = l/M*i

    if V_cl < 4:
        Pmax_j = max(dyn_pr) + Δp
        Pmin_j = max(dyn_pr) - Δp
        PPmax_J = (Pmax_j - p_extra)/(lp[-1] - lp[0])
        PPmin_J = (Pmin_j - p_extra)/(lp[-1] - lp[0])
        Pmax_Jouk = p_extra + PPmax_J*lp
        Pmin_Jouk = p_extra + PPmin_J*lp
        P = Pmax_Jouk
    else:
        Pm = 2*l*v/(g*V_cl)
        P_m = Pm
        Pmax_m = max(dyn_pr) + P_m
        Pmin_m = max(dyn_pr) - P_m
        PPmax_M = (Pmax_m - p_extra)/(lp[-1] - lp[0])
        PPmin_M = (Pmin_m - p_extra)/(lp[-1] - lp[0])
        Pmax_Mich = p_extra + PPmax_M*lp
        Pmin_Mich = p_extra + PPmin_M*lp
        P = Pmax_Mich
    blank.insert(0, round(P[-1], 3))

    fig, ax = plt.subplots()
    ls = np.linspace(0, l, M + 1)
    l1 = plt.plot(ls, P, label = 'Maximum pressure')
    l5 = plt.hlines(PN, 0, l, linestyles = '--', color = 'r', linewidth = 2.5, label = 'PN')
    plt.annotate(round(P[-1], 3), (l, P[-1]), textcoords="offset points", xytext = (0, 10), ha='right', fontsize = 8)
    legend = ax.legend(loc = 'lower left', fontsize = 9)
    fig.suptitle('Development of pressures', fontsize=15)
    fig.subplots_adjust(top=0.85)
    ax.set_title('Case of Valve Closure at ' + str(V_cl) + ' sec', size= 12)
    plt.ylabel('Pressure [m]', size = 10)
    plt.xlabel('Length [m]', size = 10)
    Y = max(P[-1], PN)
    plt.ylim(0, Y + 10)
    plt.xlim(0, l + 10)
    plt.grid(linestyle='dotted')
    plt.show()


Button(main, text='Plot', command=plot).grid(row=11, column=1, sticky=W)
Button(main, text='Calculate', command=WH).grid(row=10, column=1, sticky=W)

mainloop()