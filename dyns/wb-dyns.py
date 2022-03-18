# imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import random
import pandas as pd

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import random
import numba
from numba import jit


# Simple function for euler approximation of dyns
def run_simulation(ics, ode=None, dt=0.01, num_steps=2000, ode_ops={}):

    # initialize output vector
    num_svs = len(ics)
    ys = np.zeros((int(num_steps/dt), num_svs))
    ys[:] = np.NaN

    # Set initial values
    ys[0, :] = ics

    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(ys.shape[0]-1):

        # calculate next step
        y_dot = ode(ys[i, :], ode_ops=ode_ops)
        ys[i + 1, :] = ys[i, :] + (y_dot * dt)

        # printout on 10% progress
        #if i % (ys.shape[0]//10) == 0:
        #    print('Simulating step {}'.format(i))
    
    return ys


def plot_simulation(ys, dt=0.01, num_steps=2000):

    # set some local vars
    num_svs = ys.shape[1]
    xvals = np.arange(num_steps, step=dt)
    plot_offset = num_svs-1

    # plot overlapping line plot of each state variable
    plt.figure()
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    # iterate state variables and plot
    for sv in range(num_svs):

        # plot first plot
        to_plot = ys[:,sv]
        ax1.plot(xvals, to_plot, label='x{}'.format(sv))

        # plot second
        to_plot2 = (to_plot - np.min(to_plot)) / (np.max(to_plot) - np.min(to_plot))
        ax2.plot(xvals, to_plot2 + plot_offset, label='x{}'.format(sv))

        plot_offset -= 1
    
    # decorate ax1
    ax1.legend(loc='upper left')
    ax1.set_xlabel('t')
    ax1.set_ylabel('y')

    # decorate ax2
    ax2.legend(loc='upper left')
    ax2.set_xlabel('t')
    ax2.get_yaxis().set_visible(False)


# modle for single harmonic oscillator
def harmonic_oscillator_ode(ys_i, ode_ops={}):

    vt_old = ys_i[0]
    xt_old = ys_i[1]
    k = ode_ops['k']
    m = ode_ops['m']

    # simple harmonic oscillator as two coupled odes
    dvdt = (-k/m)*xt_old
    dxdt = vt_old

    return np.array([dvdt, dxdt])


# modle for coupled harmonic oscillators
def coupled_oscillator_ode(ys_i, ode_ops={}):

    # unpack local vars
    v1_old = ys_i[0]
    x1_old = ys_i[1]
    v2_old = ys_i[2]
    x2_old = ys_i[3]
    k = ode_ops['k']
    kprime = ode_ops['kprime']
    m = ode_ops['m']

    # two identical harmonic oscillators, coupled by an additional spring
    dv1 = (-(k + kprime) * x1_old + kprime*x2_old)/m
    dx1 = v1_old
    dv2 = (-(k + kprime) * x2_old + kprime*x1_old)/m
    dx2 = v2_old

    return np.array([dv1, dx1, dv2, dx2])


# model for system of n sets of coupled oscillators
# def multi_coupled_oscillator()



# examples
if __name__ == '__main__':

    # test single harmonic oscillator
    ics = [0, 1]
    ode_ops = {'k': 1, 'm': 1}
    num_steps = 100
    dt = 0.001
    ys = run_simulation(ics, ode=harmonic_oscillator_ode, dt=dt, num_steps=num_steps, ode_ops=ode_ops)
    plot_simulation(ys, dt=dt, num_steps=num_steps)


    # test coupled harmonic oscillator
    ics = [0, 10, 0, 3]
    ode_ops = {'k': 1, 'm': 2, 'kprime': 5}
    num_steps = 100
    dt = 0.001
    ys = run_simulation(ics, ode=coupled_oscillator_ode, dt=dt, num_steps=num_steps, ode_ops=ode_ops)
    plot_simulation(ys[:, [1,3]], dt=dt, num_steps=num_steps)