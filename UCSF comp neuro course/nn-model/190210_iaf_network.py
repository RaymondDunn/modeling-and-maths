import numpy as np
import matplotlib.pyplot as plt

"""
integrate and fire network
20 excitatory 5 inhibitory
simulate 10s of activity
show network has some properties
"""

class NetworkModel():
    def __init__(self, dt):

        # simulation params
        self.dt = dt
        self.vRe = 0
        self.vTh = 1
        self.refractory = 5 * 10**-3

        # excitatory
        self.n_exc = 20
        self.tau_m_exc = 15 * 10**-3
        self.tau1_syn_exc = 1 * 10**-3
        self.tau2_syn_exc = 3 * 10**-3
        self.mu_exc = np.random.uniform(1.1, 1.2, (1, self.n_exc))
        self.refE = np.zeros((1, self.n_exc))

        # inhibitory
        self.n_inh = 5
        self.tau_m_inh = 10 * 10**-3
        self.tau1_syn_inh = 1 * 10**-3
        self.tau2_syn_inh = 2 * 10**-3
        self.mu_inh = np.random.uniform(1, 1.05, (1, self.n_inh))
        self.refI = np.zeros((1, self.n_inh))

        # connection probabilities
        self.pEE = 0.2
        self.pEI = 0.5
        self.pIE = 0.5
        self.pII = 0.5

        # synaptic strengths
        self.jEE = 0.024
        self.jEI = -0.045
        self.jIE = 0.014
        self.jII = -0.057

        # make neuron filter vectors
        self.Fe = self.make_F(dt, self.tau1_syn_exc, self.tau2_syn_exc, self.n_exc)
        self.Fi = self.make_F(dt, self.tau1_syn_inh, self.tau2_syn_inh, self.n_inh)

        # make connectivity matrices
        self.Jee = self.make_J(self.n_exc, self.n_exc, self.pEE, self.jEE)
        self.Jei = self.make_J(self.n_exc, self.n_inh, self.pEI, self.jEI)
        self.Jie = self.make_J(self.n_inh, self.n_exc, self.pIE, self.jIE)
        self.Jii = self.make_J(self.n_inh, self.n_inh, self.pII, self.jII)

        # make voltage/spike vectors
        self.spike_e = np.zeros((1, self.n_exc))
        self.spike_i = np.zeros((1, self.n_inh))
        self.vE = np.zeros((1, self.n_exc))
        self.vI = np.zeros((1, self.n_inh))

        # lists for saving time data
        self.vI_list = []
        self.vE_list = []
        self.syn_e_list = []
        self.syn_i_list = []
        self.spike_list = []

    # neuron filter matrices
    def make_F(self, t, tau1, tau2, n):

        # make neuron filter vector
        filt = (1/(tau2 - tau1)) * (np.exp(-t/tau2) - np.exp(-t/tau1)) 
        mat = np.ones((1, n)) * filt

        return mat

    # make connectivity matrix/weight matrix
    def make_J(self, n_x, n_y, pXY, jXY):

        # make weight matrix
        mat = np.ones((n_x, n_y)) * jXY

        # spars ify matrix according to random selection
        rand_matrix = np.random.rand(n_x, n_y) < pXY

        return mat * rand_matrix

    # update voltage matrix, add bias, return spike vector
    def update_V(self, Iinput):

        # update refractory
        self.refE = self.refE - self.dt
        self.refI = self.refI - self.dt

        # fix ref negative
        self.refE[self.refE < 0] = 0
        self.refI[self.refI < 0] = 0

        # get total amount of synaptic input from exc and inh
        syn_ee = self.get_syn_input(self.Fe, self.Jee, self.spike_e, self.dt)
        syn_ie = self.get_syn_input(self.Fe, self.Jie, self.spike_e, self.dt)
        syn_ei = self.get_syn_input(self.Fi, self.Jei, self.spike_i, self.dt)
        syn_ii = self.get_syn_input(self.Fi, self.Jii, self.spike_i, self.dt)
        syn_e = np.sum(np.concatenate((syn_ee, syn_ei), axis=1), axis=1)
        syn_i = np.sum(np.concatenate((syn_ii, syn_ie), axis=1), axis=1)

        # update V with synaptic input and Iinput
        self.vE = 1/self.tau_m_exc * (self.mu_exc - self.vE) + syn_e + Iinput
        self.vI = 1/self.tau_m_inh * (self.mu_inh - self.vI) + syn_i
        self.vE_list.append(self.vE)
        self.vI_list.append(self.vI)

        # undo V updates if neuron is refractory
        self.vE[self.refE > 0] = 0
        self.vI[self.refI > 0] = 0

        # update spike
        self.spike_e = self.vE >= self.vTh
        self.spike_i = self.vI >= self.vTh

        # set refractory on spike
        self.refE[self.spike_e] = self.refractory
        self.refI[self.spike_i] = self.refractory

        # reset voltages
        self.vE[self.spike_e] = self.vRe
        self.vI[self.spike_i] = self.vRe

        # append to internal data structs
        self.syn_e_list.append(syn_e)
        self.syn_i_list.append(syn_i)
        self.spike_list.append(np.concatenate((self.spike_e, self.spike_i), axis=1))

    def get_syn_input(self, F, J, spike, dt):

        # get weight matrix times filter
        coef = F * J

        # cross with spiking
        coef = coef * spike

        return coef

# for repeatability
np.random.seed(123)

# simulation
dt = 0.1 * 10**-3
#stim = 0.07         # increased "stim" 
stim = 0.07 * np.ones((1, 20)) * (np.random.rand(20) > 0.5)
s_to_plot = 1
t_vec = np.arange(0, s_to_plot, dt)
stim_start = s_to_plot / 4
stim_end = 3*s_to_plot / 4

# create model
model = NetworkModel(dt)

for t in t_vec:

    # set stimulation
    I = 0
    if t > stim_start and t < stim_end:
        I = stim

    # update model
    model.update_V(I)

# plot
multp = 100
arr = np.array(model.spike_list).squeeze().T
t_bins = np.arange(0, s_to_plot, dt * multp)
total_n = 25

# prep output image
total_t = int(s_to_plot / dt / multp)
arr_img = np.zeros((total_n, total_t))
for n in range(0, total_n):

    # get spike indices
    st = np.where(arr[n])[0] * dt 

    # digitize and set
    dig = np.digitize(st, t_bins) - 1
    arr_img[n, dig] = 1

plt.imshow(arr_img)
plt.show()


# other analyses
a = np.array(model.vE_list).squeeze().T

"""
multp_factor = 100
arr_img = np.zeros((int(total_n*multp_factor), total_t))
for n in range(0, total_n):

    np.histogram()

    for t in range(0, total_t):
        arr_img[int(n*multp_factor):int(n+1*multp_factor), t] = arr[n,t]

"""