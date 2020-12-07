# Loosely adapted from http://www.math.mcgill.ca/gantumur/docs/reps/RyanSicilianoHH.pdf
# Which is way simpler than going straight from the HH 1952d paper man I wasted
# a lot of time on that.
# Wait the constants were on page 520 the whole time...

import numpy as np
import pylab

# ODEs
def dvdt(v, IK, INa, Il, I, Cm=0.01):
	return 1/Cm * (I - (IK + INa + Il))

def dndt(v, n):
	return An(v) * (1 - n) - (Bn(v) * (n))

def dmdt(v, m):
	return Am(v) * (1 - m) - (Bm(v) * (m))

def dhdt(v, h):
	return Ah(v) * (1 - h) - (Bh(v) * (h))

# variables w/r/t v
def An(v):
	return (0.01 * (v + 50)) / (1 - np.exp((-(v + 50)) / 10))

def Bn(v):
	return 0.125 * np.exp(-(v + 60) / 80)

def Am(v):
	return (0.1 * (v + 35)) / (1 - np.exp((-(v + 35)) / 10))

def Bm(v):
	return 4.0 * np.exp(-0.0556 * (v + 60))

def Ah(v):
	return 0.07 * np.exp(-0.05 * (v + 60))

def Bh(v):
	return 1 / (1 + np.exp(-0.1 * (v + 30)))

# functions to calculate currents
def IK(v, n, g_bar_K=0.36, EK=-72.14):
	gK = g_bar_k * (n**4)
	return gK * (v - EK)

def INa(v, m, h, g_bar_Na=1.2, ENa=55.17):
	gNa = g_bar_Na * (m**3) * h
	return gNa * (v - ENa)

def Il(v, g_bar_l=0.003, El=-49.42):
	gl = g_bar_l
	return gl * (v - El)

# for mccormic/huguenard currents
# t type
def It(v, N=2, Iinf):

	ghat = m**N * h
	I = ghat * gmax * (E - Eeq)
	# ((1 - np.exp(-t / Tm_It(v))) ** N) * np.exp(-t/Th_It(v)) * Iinf

	return 

def Tm_It(v)
	return (1 / (np.exp((v + 132) / -16.7) + np.exp((v + 16.8) / 18.2))) + 0.612

def Th_It(v)
	if v < -80:
		return np.exp((v + 467) / 66.6)
	else:
		return np.exp((v + 22) / -10.5) + 28

# rapidly inactivating K (Ia)
def Ia(v, N=4):

	# Ia1 accounts for 60% of total conductance

	return Ia1(v, N) + Ia2(v, N)

def Ia1(v, N):

	ghat = m**N * h
	I = ghat * gmax * (E - Eeq)

	return

def Ia2(v, N):

	ghat = m**N * h
	I = ghat * gmax * (E - Eeq)

	return

def Tm_Ia(v):
	return (1 / (np.exp((v + 35.8) / 19.7) + np.exp((v + 79.7) / -12.7))) + 0.37

def Th_Ia(v):
	return Th1_Ia(v) + Th2_Ia(v)

def Th1_Ia(v):
	if v < -63:
		return (1 / (np.exp((v + 46) / 5) + np.exp((v + 238) / -37.5)))
	else:
		return 19.0

def Th2_Ia(v):
	if v < -73:
		return Th1_Ia(v)
	else:
		return 60.0

# slowly inactivating K (Ik2)
def Ik2(v, N):
	return Ik2a(v, N) + Ik2b(v, N)

def Ik2a(v, N):
	return 

def Ik2b(v, N):
	return

def Tm_Ik2(v):
	return (1 / (np.exp((v + 81) / 25.6) + np.exp((v + 132) / -18.0))) + 9.9

#def Tm_Ik2a(v):
#	return (1 / (np.exp((v + 81) / 25.6) + np.exp((v + 132) / -18.0))) + 9.9

#def Tm_Ik2b(v):
#	return Tm_Ik2a(v)

def Th_Ik2(v):
	return 0.6 * Th_Ik2a(v) + 0.4 * Th_Ik2b(v)

def Th_Ik2a(v):
	return (1 / (np.exp((v + 1329) / 200) + np.exp((v + 130) / -7.1))) + 120

def Th_Ik2b(v):
	if v < -70:
		return Th_Ik2a(v)
	else:
		return 8.9

# h currents
def Ih(v, N=1):

	ghat = m**N * h
	I = ghat * gmax * (E - Eeq)

	return

def Tm_Ih(v):
	return (1 / np.exp(-14.59 - 0.086 * v) + np.exp(-1.87 + 0.0701 * v))


# activation rates
def d_rate_dt(N, Tm, Th, Iinf):


## vars for forward Euler
# time vars
dt = 0.01
ms_to_plot = 8
t = np.arange(0, ms_to_plot, dt)
I = 0.1 	# external current applied

## initialize stored vars with first values
v_array = [-60]
n_array = [An(v_array[0]) / (An(v_array[0]) + Bn(v_array[0])) ]
m_array = [Am(v_array[0]) / (Am(v_array[0]) + Bm(v_array[0])) ]
h_array = [Ah(v_array[0]) / (Ah(v_array[0]) + Bh(v_array[0])) ]

# iterate
for i in range(0, len(t) - 1):

	# calculate new m/n/h values
	n_array.append(n_array[i] + dt * dndt(v_array[i], n_array[i]))
	m_array.append(m_array[i] + dt * dmdt(v_array[i], m_array[i]))
	h_array.append(h_array[i] + dt * dhdt(v_array[i], h_array[i]))

	# calculate conductances
	#gK = g_bar_K * n_array[i]**4
	#gNa = g_bar_Na * (m_array[i]**3) * h_array[i]
	#gl = g_bar_l

	# calculate currents (conductance * driving force)
	#IK = gK * (v_array[i] - EK)
	#INa = gNa * (v_array[i] - ENa)
	#Il = gl * (v_array[i] - El)
	k = IK(v_array[i], n_array[i])
	na = INa(v_array[i], m_array[i], h_array[i])
	il = Il(v_array[i])

	# euler approx for next voltage value
	new_v = v_array[i] + dt * dvdt(v_array[i], k, na, il, I)
	v_array.append(new_v)
