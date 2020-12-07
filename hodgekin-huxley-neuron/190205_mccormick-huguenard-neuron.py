import numpy as np
import matplotlib.pyplot as plt

class IKleak():
    def __init__(self, v):

        self.g_bar_leak = 0.003
        #K_i = 4 * 10 ** -3
        #K_o = 1.5 * 10**-1
        #Eleak =  (8.314 * 308 / 96485) * np.log(K_o/K_i)
        self.Eleak = -90
        self.i_list = []
        self.g_list = []

    def get_current(self, v, dt):

        # conductance
        gleak = self.g_bar_leak
        self.g_list.append(gleak)

        # return
        i = gleak * (v - self.Eleak) * dt
        self.i_list.append(i)
        return i

class INaleak():
    def __init__(self, v):

        # lists
        self.g_list = []
        self.i_list = []

        # params
        self.g_bar_leak = 0.0014
        #Na_i = 1.8 * 10**-2
        #Na_o = 1.45 * 10**-1
        #self.Eleak = (8.314 * 308 / 96485) * np.log(Na_o/Na_i)
        self.Eleak = 45

    def get_current(self, v, dt):

        # conductance
        gleak = self.g_bar_leak
        self.g_list.append(gleak)

        # return
        i = gleak * (v - self.Eleak) * dt
        self.i_list.append(i)
        return i

class INa():
    def __init__(self, v):

        # gating variables
        self.m_list = []
        self.h_list = []
        self.i_list = []
        self.g_list = []

        # initial conditions
        self.g_bar_Na = 0.4
        #Na_i = 1.8 * 10**-2
        #Na_o = 1.45 * 10**-1
        #self.Eleak = (8.314 * 308 / 96485) * np.log(Na_o/Na_i)
        self.ENa = 45 
        self.m_list.append(self.m_inf(v))
        self.h_list.append(self.h_inf(v))

    def m_inf(self, v):
        return self.Am(v) / (self.Am(v) + self.Bm(v))

    def h_inf(self, v):
        return self.Ah(v) / (self.Ah(v) + self.Bh(v))

    def dmdt(self, v, m):
        return self.Am(v) * (1 - m) - (self.Bm(v) * m)

    def dhdt(self, v, h):
        return self.Ah(v) * (1 - h) - (self.Bh(v) * h)

    def Am(self, v):
        #return (0.091 * (v + 38)) / (1 - np.exp(-(v + 38) / 5)) # m and h
        return (0.1 * (v + 35)) / (1 - np.exp((-(v + 35)) / 10)) # h and h


    def Bm(self, v):
        #return (-0.062 * (v + 38)) / (1 - np.exp((v + 38) / 5))  # m and h
        return 4.0 * np.exp(-0.0556 * (v + 60))  # h and h


    def Ah(self, v):
        #return 0.016 * np.exp((-55 - v) / 15)   # m and h
        return 0.07 * np.exp(-0.05 * (v + 60))   # h and h


    def Bh(self, v):
        #return 2.07 / ((np.exp(17 - v) / 21) + 1)    # m and h
        return 1 / (1 + np.exp(-0.1 * (v + 30)))      # h and h


    def get_current(self, v, dt):

        # get last values of m and h
        m = self.m_list[-1]
        h = self.h_list[-1]

        # update 
        self.m_list.append(m + dt * self.dmdt(v, m))
        self.h_list.append(h + dt * self.dhdt(v, h))

        # use new values
        m = self.m_list[-1]
        h = self.h_list[-1]

        # conductance
        gNa = self.g_bar_Na * (m**3) * h
        self.g_list.append(gNa)

        # return
        i = gNa * (v - self.ENa) * dt
        self.i_list.append(i)
        return i 

class IK():
    def __init__(self, v):

        # initial conditions
        self.g_bar_K = 0.3
        #K_i = 4 * 10 ** -3
        #K_o = 1.5 * 10**-1
        #Eleak =  (8.314 * 308 / 96485) * np.log(K_o/K_i)
        self.EK = -90
        self.i_list = []
        self.m_list = []
        self.h_list = []
        self.m_list.append(self.m_inf(v))
        self.g_list = []


    def m_inf(self, v):
        return self.Am(v) / (self.Am(v) + self.Bm(v))

    # change in gating variables 
    def dmdt(self, v, m):
        return self.Am(v) * (1 - m) - (self.Bm(v) * m)

    # forward rate constant
    def Am(self, v):
        return (0.01 * (v + 50)) / (1 - np.exp((-(v + 50)) / 10))

    # reverse rate constant
    def Bm(self, v):
        return 0.125 * np.exp(-(v + 60) / 80)

    def get_current(self, v, dt):

        # get gate
        m = self.m_list[-1]

        # update new n value
        self.m_list.append(m + dt * self.dmdt(v, m))

        # use new gate value
        m = self.m_list[-1]
        self.m_list.append(m)

        # calculate conductance
        gK = self.g_bar_K * (m**4)
        self.g_list.append(gK)

        # return current
        i = gK * (v - self.EK) * dt
        self.i_list.append(i)
        return i

class It():
    def __init__(self, v):
         
        self.g_max = 0.003
        self.N = 2
        self.v_half_m = -57
        self.v_half_h = -81 
        self.k_m = -6.2
        self.k_h = 4.0
        self.m_list = [self.m_inf(v)]
        self.h_list = [self.h_inf(v)]
        self.i_list = []
        self.g_list = []
        self.q10 = 2.5

    def Tm(self, v):
        return ((1 / (np.exp((v + 132) / -16.7) + np.exp((v + 16.8) / 18.2))) + 0.612)

    def Th(self, v):
        if v < -80:
            return np.exp((v + 467) / 66.6)
        else:
            return np.exp((v + 22) / -10.5) + 28

    def dm(self, v, m, dt):
        return self.m_inf(v) - (self.m_inf(v) - m) * np.exp(-dt / self.Tm(v))

    def dh(self, v, h, dt):
        return self.h_inf(v) - (self.h_inf(v) - h) * np.exp(-dt / self.Th(v))

    def m_inf(self, v):
        return (1 / (1 + np.exp((v - self.v_half_m )/ self.k_m))) ** self.N

    def h_inf(self, v):
        return (1 / (1 + np.exp((v - self.v_half_h )/ self.k_h))) ** self.N

    def get_current(self, v, dt):
        
        # update gating variables
        m = self.m_list[-1]
        h = self.h_list[-1]

        # calc changes
        m = self.dm(v, m, dt)
        h = self.dh(v, h, dt)

        # add to list
        self.m_list.append(m)
        self.h_list.append(h)

        # calculate current
        E = v / 1000
        F = 96485
        R = 8.314
        T = 35.5 + 273.15
        z = 2
        ca_out = 3*10**-3  # mM
        ca_in = 10*10**-9  # nm
        P = self.g_max

        # calculate g_hat
        g_hat = (m ** self.N) * h
        self.g_list.append(g_hat)

        # return current
        i = g_hat *  P * (z**2) * E * (F ** 2) / (R * T) * (ca_in - ca_out * np.exp((-z * F * E) / (R * T)) / (1 - np.exp((-z * F * E) / (R * T)))) * dt
        self.i_list.append(i)
        return i

# initial settings for simulation
dt = 1 * 10**-3
ms_to_plot = 50
stim_start = ms_to_plot / 4
stim_end = 3*ms_to_plot / 4
t = np.arange(0, ms_to_plot, dt)
stim = 0.2
initial_v = -64
Cm = 0.003
v_list = [initial_v]

# initialize our channel objects
#k = IK(initial_v)
na = INa(initial_v)
kleak = IKleak(initial_v)
k = IK(initial_v)
it = It(initial_v)
naleak = INaleak(initial_v)

# iterate timepoints
channels = [na, kleak, k, naleak]
#channels = [it, kleak, naleak]
for i in range(0, len(t) - 1):

    I = 0
    if i*dt > stim_start and i*dt < stim_end:
        I = stim

    # get last v
    v = v_list[-1]
    #dvdt = 1 / Cm * (I*dt - (leak.get_current(v, dt) + na.get_current(v, dt)))
    #dvdt = 1 / Cm * (I*dt - (kleak.get_current(v, dt) + na.get_current(v, dt) + k2.get_current(v, dt)))
    #it.get_current(v, dt)
    Iions = 0
    for c in channels:
        Iions += c.get_current(v, dt)

    dvdt = 1 / Cm * (I*dt - Iions)
    v_list.append(v + dvdt)

plt.plot(t, v_list)
plt.xlabel('ms')
plt.ylabel('Vm')
plt.show()


# plot info for channel
"""
c = kleak
c = na
c = k
c = it
plt.plot(c.g_list)
plt.plot(c.i_list)
plt.legend(['g', 'i'])
plt.show()


c = na
c = k
c = it
plt.plot(c.m_list)
plt.plot(c.h_list)
plt.legend(['m', 'h'])
plt.show()


plt.plot(kleak.i_list)
plt.plot(na.i_list)
plt.plot(k.i_list)
plt.plot(it.i_list)
plt.legend(['kleak', 'na', 'k', 'it'])
plt.show()


# plot as a function of voltage
vmin = -120
vmax = 80
xvals = np.arange(vmin, vmax, 1)
c_v = It(vmin)
hinf = [c_v.h_inf(vmin)]
minf = [c_v.m_inf(vmin)]
for v in range(vmin, vmax-1):
    c_v.get_current(v, dt)
    hinf.append(c_v.h_inf(v))
    minf.append(c_v.m_inf(v))
plt.plot(xvals, c_v.m_list)
plt.plot(xvals, c_v.h_list)
plt.plot(xvals, hinf)
plt.plot(xvals, minf)
plt.legend(['m', 'h', 'h_inf', 'minf'])
plt.show()

"""