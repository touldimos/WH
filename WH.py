#WATER HAMMER
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pandas as pd
from tabulate import tabulate

def regcond(D, e, Qus, k):
    Din = D - 2*e 
    A = (np.pi*(Din/1000)**2)*0.25
    v = Qus/1000/A
    f0 = (1/(2*np.log10(k/(3.72*Din))))**2
    initial = Din, A, v, f0
    return initial

def losses(f, l, v, Din, g, local_losses):
    dh = f*l*v**2/(Din/1000*2*g)
    dh_loc = local_losses/100*dh
    DH = dh + dh_loc
    return DH

def gr(zds, zus, l, M, N, DH, p_extra):
    slope = (zds - zus)/l
    z = np.zeros(M + 1)
    for i in range(0, M + 1, N):
        z[0] = zus
        z[i] = z[i-1] + l/M*slope
        ground = z
    Pz_us = zus + p_extra
    Pz_ds = Pz_us - DH
    j = DH/l
    Pz = np.zeros(M + 1)
    for i in range(0, M + 1, N):
        Pz[0] = Pz_us
        Pz[i] = Pz[i - 1] - j*l/M
        table3 = ground, Pz_us, Pz_ds, Pz
    return table3

def DP(K, ρ, D, μ, E, e, l, v, g, ground):
    a = np.sqrt(K/ρ)/(np.sqrt(1 + (K*(D/1000)*(1 - μ**2))/(E*e/1000)))
    t = 2*l/a
    Δp = a*v/g
    ΔP = ground + Δp
    table2 = Δp, ΔP, t   
    return table2
   
def jouk(dyn_pr, Δp):
    Pmax_j = max(dyn_pr) + Δp
    Pmin_j = max(dyn_pr) - Δp
    table = Pmax_j, Pmin_j
    return table

def mich(l, v, g, V_cl, dyn_pr):
    Pm = 2*l*v/(g*V_cl)
    P_m = Pm
    Pmax_m = max(dyn_pr) + P_m
    Pmin_m = max(dyn_pr) - P_m
    table = Pmax_m, Pmin_m
    return table

def Hammer(M, N, l, Pmax_j, Pmin_j, p_extra, Pmax_m, Pmin_m):
    lp = np.zeros(M + 1)
    for i in range(0, M + 1, N):
        lp[0] = 0
        lp[i] = l/M*i
    
    PPmax_J = (Pmax_j - p_extra)/(lp[-1] - lp[0])
    PPmin_J = (Pmin_j - p_extra)/(lp[-1] - lp[0])
    Pmax_Jouk = p_extra + PPmax_J*lp
    Pmin_Jouk = p_extra + PPmin_J*lp

    PPmax_M = (Pmax_m - p_extra)/(lp[-1] - lp[0])
    PPmin_M = (Pmin_m - p_extra)/(lp[-1] - lp[0])
    Pmax_Mich = p_extra + PPmax_M*lp
    Pmin_Mich = p_extra + PPmin_M*lp

    Pressures = Pmax_Jouk, Pmin_Jouk, Pmax_Mich, Pmin_Mich
    return Pressures

def print_table(Pmax_Jouk, Pmin_Jouk, Pmax_Mich, Pmin_Mich):
    df = pd.DataFrame({'Sudden Close \n Maximum Pressure \n [m]':Pmax_Jouk, 
                    'Sudden Close \n Minimum Pressure \n [m]':Pmin_Jouk, 
                    'Slow Close \n Maximum Pressure \n [m]':Pmax_Mich,
                    'Slow Close \n Minimum Pressure \n [m]':Pmin_Mich})
    df2 = df.round(2)
    return tabulate(df2, headers="keys", tablefmt="simple", numalign="decimal", showindex = False)

def show_fig(Pmax_Jouk, Pmin_Jouk, Pmax_Mich, Pmin_Mich, V_cl, l, M, PN):
    plt.rc('text', usetex=True)  
    plt.rc('font', family='serif')
    fig, ax = plt.subplots()
    ls = np.linspace(0, l, M + 1)
    l1 = plt.plot(ls, Pmax_Jouk, label = 'Max Sudden closure')
    l2 = plt.plot(ls, Pmin_Jouk, label = 'Min Sudden closure')
    l3 = plt.plot(ls, Pmax_Mich, label = 'Max Slow closure')
    l4 = plt.plot(ls, Pmin_Mich, label ='Min Slow closure')
    l5 = plt.hlines(PN, 0, l, linestyles = '--', color = 'r', linewidth = 2.5, label = 'PN')
    plt.annotate(round(Pmax_Jouk[-1], 3), (l, Pmax_Jouk[-1]), textcoords="offset points", xytext=(0, 10), ha='right')
    plt.annotate(round(Pmin_Jouk[-1], 3), (l, Pmin_Jouk[-1]), textcoords="offset points", xytext=(0, 10), ha='right')
    legend = ax.legend(loc = 'lower left')
    fig.suptitle(r'\textbf{Development of pressures}', fontsize=18)
    fig.subplots_adjust(top=0.85)
    ax.set_title('Case of Sudden Closure and Slow Valve Closure at ' + str(V_cl) + ' sec', size= 15)
    plt.ylabel('Pressure [m]', size = 12)
    plt.xlabel('Length [m]', size = 12)
    plt.ylim(0, PN + 10 )
    plt.xlim(0, l + 10)
    plt.grid(linestyle='dotted')
    plt.show()

def Check(Pmax_Jouk, PN):
    if Pmax_Jouk[-1] < PN:
        print(" ")
        print("Pipe is sufficient. The maximum water hammer pressure increased of the static is", round(Pmax_Jouk[-1], 3), "m <", PN, "m")
    else:
        print(" ")
        print("Pipe is not sufficient. The maximum water hammer pressure increased of the static is", round(Pmax_Jouk[-1], 3), "m >", PN, "m")
        print(" ")
    return