import pylab


'''
Using your favorite programming environment, simulate and plot the time course of the voltage of a spiking Hodgkin Huxley neuron so you can see the shape of a few consecutive spikes.  Also make a plot of firing rate versus injected current. What’s the current threshold to spike?

Hints:

-Use the differential equations on page 518/519 of (Hodgkin & Huxley 1952d, linked on the lecture materials website).  Remember that you'll want dV/dt on the left hand side and everything else on the right hand side.
-Euler’s method should be sufficient to code your simulation, just make sure you use a small enough time step. (http://faculty.olin.edu/bstorey/Notes/DiffEq.pdf)
-If you are feeling fancy you can use the Runge-Kutta method.


'''
# I = Cm * dVdt + gK * n**4(V - Vk) + gNa * (m**3)*h(V - VNa) + gl(V - Vl)
# I - Cm * dVdt =  gK * n**4(V - Vk) + gNa * (m**3)*h(V - VNa) + gl(V - Vl)

"""
Icapacity = Cm * dVdt
IK = gK * (n**4)(V - Vk)
INa = gNa * (m**3)*h(V - VNa)
Ileak = gl(V - Vl)
"""

# constants (pg 520)
Cm = 1.0
VNa = -115
Vk = 12
Vl = 10.613
gNa = 120
gK = 36
gl = 0.3

# pg 518/19
An = 0.01 * (V + 10) / (pylab.exp((V + 10) / 10) - 1)
Bn = 0.125 * pylab.exp(v/80)
Am = 0.1 * (V + 25) / (pylab.exp((V + 25) / 10) - 1)
Bm = 4 * pylab.exp(V / 18)
Ah = 0.07 * pylab.exp(V / 20)
Bh = 1 / (pylab.exp((V + 30)/ 10) + 1)

# Initial conditions
"""
 If the stimulus is a short shock at t = 0, the form of the 
 action potential should be given by solving eqn. (26) with I = 0 
 and the initial conditions that V = Vo and
m, n and h have their resting steady state values, when t =0.
"""
I = 0
t = 0
V = V0

# pg 507
# n = n_inf - (n_inf - n0) * pylab.exp(-t/Tn)
# n_inf = An / (An + Bn)
# Tn = 1 / (An + Bn)
n0 = An / (An / Bn)

# pg 512
# m = m0
# m = m_inf - (m_inf - m0) * pylab.exp(-t/Tm)
# m_inf = Am / (Am / Bm)
# Tm = 1/(Am + Bm)
m0 = Am / (Am / Bm)

# h = h0
# h = h_inf - (h_inf - h0) * pylab.exp(-t/Th)
# h_inf = Ah / (Ah / Bh)
# Th = 1/(Ah + Bh)
h0 = Ah / (Ah / Bh)

# pg 518/19
dVdt = (gK * n**4(V - Vk) + gNa * (m**3)*h(V - VNa) + gl(V - Vl) - I) / -1*(Cm)
dndt = An*(1-n) - (Bn*n)
dmdt = Am*(1-m) - (Bm*m)
dhdt = Ah(1-h) - (Bh*h)


#################

# Initial conditions and constants
V = 0
I = 0
t = 0
Cm = 1.0
VNa = -115
Vk = 12
Vl = 10.613
gNa = 120
gK = 36
gl = 0.3
An = 0.01 * (V + 10) / (pylab.exp((V + 10) / 10) - 1)
Bn = 0.125 * pylab.exp(V / 80)
Am = 0.1 * (V + 25) / (pylab.exp((V + 25) / 10) - 1)
Bm = 4 * pylab.exp(V / 18)
Ah = 0.07 * pylab.exp(V / 20)
Bh = 1 / (pylab.exp((V + 30)/ 10) + 1)
n0 = An / (An / Bn)
m0 = Am / (Am / Bm)
h0 = Ah / (Ah / Bh)

# Vars
stepSize = 0.01
n = n0
m = m0
h = h0
secondsToSimulate = .5

# iterate
toPlot = []
n_array = []
m_array = []
h_array = []
step_array = pylab.arange(0, secondsToSimulate, stepSize)

for s in step_array:

	# store
	toPlot.append(V)
	n_array.append(n)
	m_array.append(m)
	h_array.append(h)

	# calculate new membrane voltage
	dVdt = (gK * (n**4)*(V - Vk) + gNa * (m**3)*h*(V - VNa) + gl*(V - Vl) - I) / -1*(Cm)
	V = V + dVdt

	# calculate changes in alphas/betas (pg 519) based on voltage
	An = 0.01 * (V + 10) / (pylab.exp((V + 10) / 10) - 1)
	Bn = 0.125 * pylab.exp(V / 80)
	Am = 0.1 * (V + 25) / (pylab.exp((V + 25) / 10) - 1)
	Bm = 4 * pylab.exp(V / 18)
	Ah = 0.07 * pylab.exp(V / 20)
	Bh = 1 / (pylab.exp((V + 30)/ 10) + 1)
	dndt = An*(1-n) - (Bn*n)
	dmdt = Am*(1-m) - (Bm*m)
	dhdt = Ah*(1-h) - (Bh*h)

	# Calculate changes in driving force


	# update vars
	V = V + dVdt
	n = n + dndt
	m = m + dmdt
	h = h + dhdt

# make figure
fig = pylab.figure()

# plot figure


# show figure
pylab.show()



########################
# Attempt 2

# we need to calculate, for each step:
# New driving forces for each conductance
# Change in membrane voltage (change in sum of each conductance)

# I = Cm * dVdt + gK * n**4(V - Vk) + gNa * (m**3)*h(V - VNa) + gl(V - Vl)
# I - (gK * n**4(V - Vk) + gNa * (m**3)*h(V - VNa) + gl(V - Vl)) = Cm * dVdt	# Cm is 1?


"""
n0 = An / (An + Bn)
m0 = Am / (Am + Bm)
h0 = Ah / (Ah + Bh)

I = 0



dndt = An * (1 - n) - (Bn * n)
dmdt = Am * (1 - m) - (Bm * m)
dhdt = Ah * (1 - h) - (Bh * h)
dVdt = I - (gK * n**4(V - Vk) + gNa * (m**3)*h(V - VNa) + gl(V - Vl))


An = 0.01 * (V + 10) / (pylab.exp((V + 10) / 10) - 1)
Bn = 0.125 * pylab.exp(V / 80)
Am = 0.1 * (V + 25) / (pylab.exp((V + 25) / 10) - 1)
Bm = 4 * pylab.exp(V / 18)
Ah = 0.07 * pylab.exp(V / 20)
Bh = 1 / (pylab.exp((v + 30) / 10) + 1)
"""


# constants
gNa = 120
gK = 36
gl = 0.3

Vk = 12
VNa = -115
Vl = 10.613

# iterate in steps
# Vars
stepSize = 0.01
secondsToSimulate = 1
step_array = pylab.arange(0, secondsToSimulate, stepSize)

# initial variables
I = 0
V = 0
n = n0
m = m0
h = h0

# arrays to store intermediate variables 
toPlot = []
n_array = []
m_array = []
h_array = []

# iterate steps
for step in step_array:

	# store vars
	toPlot.append(V)
	n_array.append(n)
	m_array.append(m)
	h_array.append(h)

	# update voltage
	dVdt = I - (gK * (n**4)*(V - Vk) + gNa * (m**3)*(h * (V - VNa)) + gl * (V - Vl))
	V = V + dVdt

	# use voltage to update driving forces
	An = 0.01 * (V + 10) / (pylab.exp((V + 10) / 10) - 1)
	Bn = 0.125 * pylab.exp(V / 80)
	Am = 0.1 * (V + 25) / (pylab.exp((V + 25) / 10) - 1)
	Bm = 4 * pylab.exp(V / 18)
	Ah = 0.07 * pylab.exp(V / 20)
	Bh = 1 / (pylab.exp((V + 30) / 10) + 1)

	# calculate change in driving forces
	dndt = An * (1 - n) - (Bn * n)
	dmdt = Am * (1 - m) - (Bm * m)
	dhdt = Ah * (1 - h) - (Bh * h)

	# update driving force
	n = n + dndt
	m = m + dmdt
	h = h + dhdt

