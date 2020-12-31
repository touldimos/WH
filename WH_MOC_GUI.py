from tkinter import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from ipywidgets import interact, interactive, fixed, interact_manual

main = Tk()
# main.geometry('880x250')
Label(main, text = "Properties of the fluid", font=("Helvetica", 8, 'bold')).grid(row=0)

Label(main, text = "Fluid's viscosity (V) [m\u00b2/s]:", font=("Helvetica", 8)).grid(row=1)
Label(main, text = "Pipe's elasticity (E) [Pa]:", font=("Helvetica", 8)).grid(row=2)
Label(main, text = "Fluid's elasticity (Kf) [Pa]:", font=("Helvetica", 8)).grid(row=3)
Label(main, text = "Fluid's density (rho) [kg/m\u00b3]:", font=("Helvetica", 8)).grid(row=4)
Label(main, text = "Pipe's freedom (mi):", font=("Helvetica", 8)).grid(row=5)

Label(main, text = "Network", font=("Helvetica", 8, 'bold')).grid(row=0, column=3)
Label(main, text = "Elevation US [m]:", font=("Helvetica", 8)).grid(row=1, column=3)
Label(main, text = "Elevation DS [m]:", font=("Helvetica", 8)).grid(row=2, column=3)
Label(main, text = "Discharge (Q) [m\u00b3/s]:", font=("Helvetica", 8)).grid(row=3, column=3)
Label(main, text = "Diameter(D) [mm]:", font=("Helvetica", 8)).grid(row=4, column=3)
Label(main, text = "Pipe's Length (L) [m]:", font=("Helvetica", 8)).grid(row=5, column=3)
Label(main, text = "Pipe's Roughness (k) [mm]", font=("Helvetica", 8)).grid(row=6, column=3)
Label(main, text = "Wall thickness (e) [mm]:", font=("Helvetica", 8)).grid(row=7, column=3)
Label(main, text = "Local losses [%]:", font=("Helvetica", 8)).grid(row=8, column=3)

Label(main, text = "Transient analysis", font=("Helvetica", 8, 'bold')).grid(row=0, column=5)
Label(main,  text = "Reservoir (Hzero) [m]:", font=("Helvetica", 8)).grid(row=1, column=5)
Label(main, text = "Valve closure (tclose) [s]:", font=("Helvetica", 8)).grid(row=2, column=5)
Label(main, text = "Interior nodes [-]:", font=("Helvetica", 8)).grid(row=3, column=5)
Label(main, text = "Tmax [sec]:", font=("Helvetica", 8)).grid(row=4, column=5)
Label(main, text = "Node to plot:", font=("Helvetica", 8)).grid(row=5, column=5)
Label(main, text = "Number of time interval:", font=("Helvetica", 8)).grid(row=6, column=5)

Label(main, text = "Results", font=("Helvetica", 8, 'bold')).grid(row=9, column=5)
Label(main, text = "Maximum Head (Hmax) [m]:", font=("Helvetica", 8)).grid(row=10, column=5)
Label(main, text = "Minimum Head (Hmax) [m]:", font=("Helvetica", 8)).grid(row=11, column=5)

CheckVar1 = IntVar()
CheckVar2 = IntVar()
# CheckVar3 = IntVar()

C1 = Checkbutton(main, text = "Jouk", variable = CheckVar1, \
                 onvalue = 1, offvalue = 0, height=1, \
                 width = 20)
C2 = Checkbutton(main, text = "Mich", variable = CheckVar2, \
                 onvalue = 1, offvalue = 0, height=1, \
                 width = 20)
# C3 = Checkbutton(main, text = "Export Head", variable = CheckVar3, \
#                  onvalue = 1, offvalue = 0, height=1, \
#                  width = 20)

C1.grid(row=7, column=5)
C2.grid(row=7, column=6)
# C3.grid(row=10, column=2)
num1 = Entry(main)
num1.insert(0, '{:2e}'.format(1.01*10**-6))
num2 = Entry(main)
num2.insert(0, '{:2e}'.format(784532117.25))
num3 = Entry(main)
num3.insert(0, '{:2e}'.format(1.96*10**9))
num4 = Entry(main)
num4.insert(0, 999.4)
num5 = Entry(main)
num5.insert(0, 0.3)
num6 = Entry(main)
num6.insert(0, 30)
num7 = Entry(main)
num7.insert(0, 10)
num8 = Entry(main)
num8.insert(0, 1.988)
num9 = Entry(main)
num9.insert(0, 0.84)
num10 = Entry(main)
num10.insert(0, 1000)
num11 = Entry(main)
num11.insert(0, 0.01)
num12 = Entry(main)
num12.insert(0, 0.07)
num13 = Entry(main)
num13.insert(0, 10)
num14 = Entry(main)
num14.insert(0, 120)
num15 = Entry(main)
num15.insert(0, 0)
num16 = Entry(main)
num16.insert(0, 11)
num17 = Entry(main)
num17.insert(0, 50)
num18 = Entry(main)
num18.insert(0, 10)
num19 = Entry(main)
num19.insert(0, 5)
blank = Entry(main)
blank1 = Entry(main)

num1.grid(row=1, column=2)
num2.grid(row=2, column=2)
num3.grid(row=3, column=2)
num4.grid(row=4, column=2)
num5.grid(row=5, column=2)
num6.grid(row=1, column=4)
num7.grid(row=2, column=4)
num8.grid(row=3, column=4)
num9.grid(row=4, column=4)
num10.grid(row=5, column=4)
num11.grid(row=6, column=4)
num12.grid(row=7, column=4)
num13.grid(row=8, column=4)
num14.grid(row=1, column=6)
num15.grid(row=2, column=6)
num16.grid(row=3, column=6)
num17.grid(row=4, column=6)
num18.grid(row=5, column=6)
num19.grid(row=6, column=6)
blank.grid(row=10, column=6)
blank1.grid(row=11, column=6)

def hammer():
    n = float(num1.get())                #Viscosity [m2/s]
    E = float(num2.get())*100                #Pipe's elasticity [Pa]
    Kf = float(num3.get())               #Fluid's elasticity [Pa]
    rho = float(num4.get())              #Fluid's density [kg/m3]
    mi = float(num5.get())               #Pipe's freedom
    
    #Initial conditions
    elevd = float(num6.get())
    elevup = float(num7.get())
    Q = float(num8.get())
    D = float(num9.get())
    l = float(num10.get())
    K = float(num11.get())                           #Pipe roughness [mm]
    e = float(num12.get())
    hfl = float(num13.get())
    hzero = float(num14.get())
    tclose = float(num15.get())
    nodes = int(num16.get())
    tmax = float(num17.get())
    node_to_plot = int(num18.get())
    time_to_plot = int(num19.get())
    g = 9.81
    blank.delete(0, END)
    blank1.delete(0, END)
    
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
    
    hf = f*l*v**2/(Din*2*g)
    hl = hfl/100*hf
    dh = hf + hl
    
    a = np.sqrt(Kf/rho)/(np.sqrt(1 + (Kf * D * (1 - mi**2))/(E * e)))
    dx = l/(nodes - 1)
    dt = dx/a
    dp = a * v / g
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
    
    def jouk(h, dp):
        Pmax_j = max(h) + dp
        Pmin_j = max(h) - dp
        return Pmax_j, Pmin_j
    
    def mich(l, v, g, tclose, h):
        Pm = 2*l*v/(g*tclose)
        Pmax_m = max(h) + Pm
        Pmin_m = max(h) - Pm
        return Pmax_m, Pmin_m
    
    def chars(nodes, Q, D, n, K, H, ca, dt, hzero, V0, t, tclose, A):
        def fcw(n, D, v, K):
            midf = []
            Re = v * D / n
            f = (1/(2*np.log10(K/(3.72 * D * 1000))))**2
            midf.append(f)
            for i in range(1, 10):
                midf.append((1 / (2 * np.log10((K / (3.72 * D * 1000)) + (2.51/(Re * np.sqrt(midf[i - 1]))))))**2)
            f = midf[-1]
            return f
        cm_ = np.zeros(nodes)
        cn_ = np.zeros(nodes)
        Hnew = np.zeros(nodes)
        Qnew = np.zeros(nodes)
        Vnew = np.zeros(nodes)
        for i in range(1, nodes):
            if Q[i - 1] != 0:
                vi = Q[i - 1] / A
                f = fcw(n, D, np.abs(vi), K)
            else:
                f = 0
            cfa = f * dt / (np.pi * D ** 3/2)
            cm = Q[i - 1] + ca * H[i - 1] - cfa * Q[i - 1] * abs(Q[i - 1])
            cm_[i] = cm
        for i in range(0, nodes - 1):
            if Q[i + 1] != 0:
                vi = Q[i + 1] / A
                f = fcw(n, D, np.abs(vi), K)
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
    
    H.append(chars(nodes, Qic, Din, n, K, Hic, ca, dt, hzero, v, dt, tclose, A)[0])
    Q.append(chars(nodes, Qic, Din, n, K, Hic, ca, dt, hzero, v, dt, tclose, A)[1])
    V.append(chars(nodes, Qic, Din, n, K, Hic, ca, dt, hzero, v, dt, tclose, A)[2])
    
    for i in range(1, len(step)):
        H.append(chars(nodes, Q[i], Din, n, K, H[i], ca, dt, hzero, v, step[i], tclose, A)[0])
        Q.append(chars(nodes, Q[i], Din, n, K, H[i], ca, dt, hzero, v, step[i], tclose, A)[1])
        V.append(chars(nodes, Q[i], Din, n, K, H[i], ca, dt, hzero, v, step[i], tclose, A)[2])
    
    Head = pd.DataFrame(H)
    Discharge = pd.DataFrame(Q)
    Velocity = pd.DataFrame(V)
    
    blank.insert(0, np.round(Head[10].max(), 4))
    blank1.insert(0, np.round(Head[10].min(), 4))
    
    
    # if CheckVar3.get() == True:
    #     Head.astype(float).round(3).to_excel('transient.xlsx')
        
    time = Head.index * dt
    
    plt.figure()
    plt.plot(time, Head[node_to_plot], 'b', zorder = 11)
    plt.vlines(tclose, -abs(Head[10].min()), Head[10].max(), 'r', ls = '--', alpha = 0.75, zorder = 10)
    plt.hlines(pipez[node_to_plot], 0, time[-1], 'darkgreen', alpha = 0.8, zorder = 8)
    plt.hlines(0, -5, time[-1], 'black', zorder = 9)
    plt.vlines(-5, -abs(Head[10].min()), Head[10].max(), 'black', zorder = 8)
    plt.legend(['Pressure wave', 'Valve close {} secs'.format(tclose), 'Ground level'])
    plt.title('Node {}'.format(node_to_plot))
    plt.xlabel('Time [sec]')
    plt.ylabel('Head [m]')
    plt.grid()
    plt.show();
    
    jou = CheckVar1.get()
    mic = CheckVar1.get()
    
    timming = time_to_plot * dt
    
    plt.figure()
    plt.plot(Head.loc[time_to_plot], 'b', zorder = 10)
    plt.plot(pipez, 'darkgreen', alpha = 0.8, zorder = 10)
    plt.plot(Head.max(axis = 0), 'r--')
    plt.plot(Head.min(axis = 0), 'y--')
    plt.hlines(0, 0, nodes - 1, 'black', zorder = 9)
    plt.vlines(0, -abs(Head[10].min()), Head[10].max(), 'black', zorder = 8)
    plt.legend(['Pressure wave', 'Ground level', 'Maximum', 'Minimum'])
    if tclose > 0 and mic and jou:
        plt.hlines(mich(l, v, g, tclose, h)[0], 0, nodes - 1)
        plt.hlines(mich(l, v, g, tclose, h)[1], 0, nodes - 1)
        plt.hlines(jouk(h, dp)[0], 0, nodes - 1)
        plt.hlines(jouk(h, dp)[1], 0, nodes - 1)
        plt.legend(['Pressure wave', 'Ground level', 'Maximum', 'Minimum', 'Joukowski max', 'Joukowski max', 'Michaud max', 'Michaud min'])
    elif tclose > 10 and mic and not jou:
        plt.hlines(mich(l, v, g, tclose, h)[0], 0, nodes - 1)
        plt.hlines(mich(l, v, g, tclose, h)[1], 0, nodes - 1)
        plt.legend(['Pressure wave', 'Ground level', 'Maximum', 'Minimum', 'Michaud max', 'Michaud min'])
    elif jou:
        plt.hlines(jouk(h, dp)[0], 0, nodes - 1)
        plt.hlines(jouk(h, dp)[1], 0, nodes - 1)
        plt.legend(['Pressure wave', 'Ground level', 'Maximum', 'Minimum', 'Joukowski max', 'Joukowski max'])
            
    plt.title('Time {:.3f} sec'.format(timming * dt))
    plt.xlabel('Nodes [-]')
    plt.ylabel('Head [m]')
    plt.grid()
    plt.show();

        
Button(main, text='Calculate', command=hammer).grid(row=10, column=1, sticky=W)

mainloop()
