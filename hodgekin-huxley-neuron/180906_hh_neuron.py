# Loosely adapted from http://www.math.mcgill.ca/gantumur/docs/reps/RyanSicilianoHH.pdf
# Which is way simpler than going straight from the HH 1952d paper man I wasted
# a lot of time on that.
# Wait the constants were on page 520 of HH1952d the whole time... dammit ok

import numpy as np
import pylab

# ODEs
def dvdt(v, IK, INa, Il):
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

# constants
Cm = 0.01
ENa = 55.17
EK = -72.14
El = -49.42
g_bar_Na = 1.2
g_bar_K = 0.36
g_bar_l = 0.003

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
	gK = g_bar_K * n_array[i]**4
	gNa = g_bar_Na * (m_array[i]**3) * h_array[i]
	gl = g_bar_l

	# calculate currents (conductance * driving force)
	IK = gK * (v_array[i] - EK)
	INa = gNa * (v_array[i] - ENa)
	Il = gl * (v_array[i] - El)

	# euler approx for next voltage value
	new_v = v_array[i] + dt * dvdt(v_array[i], IK, INa, Il)
	v_array.append(new_v)
