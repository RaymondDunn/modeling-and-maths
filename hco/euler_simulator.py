### hard coded HCO_fast can provide extermal input to main HCO
# magnitude of this input is on a slider
# no sliders or visualization for the parameters of HCO_fast

import numpy as np
import time as time
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
from numba import jit
import json
import os
from datetime import datetime
import matplotlib.gridspec as gridspec

def euler_simulator(ode_system, initial_conditions, t_span, dt):
    """
    Run an Euler simulation of a system of ODEs.
    
    Parameters:
    -----------
    ode_system : callable
        Function that takes (t, y) as arguments and returns the derivatives dy/dt
        The function should return an array of derivatives matching the dimension of initial_conditions
    initial_conditions : array-like
        Initial values of the state variables
    t_span : tuple
        (t_start, t_end) - the time interval to simulate
    dt : float
        Time step for the simulation
    
    Returns:
    --------
    t : array
        Time points of the simulation
    y : array
        Solution array where each row corresponds to a time point and each column to a state variable
    """
    # Create time array
    t = np.arange(t_span[0], t_span[1] + dt, dt)
    n_steps = len(t)
    
    # Initialize solution array
    y = np.zeros((n_steps, len(initial_conditions)))
    y[0] = initial_conditions
    
    # Run Euler integration
    for i in range(1, n_steps):
        # Calculate derivatives at current point
        dy = ode_system(t[i-1], y[i-1])
        
        # Update state variables using Euler method
        y[i] = y[i-1] + dt * dy
    
    return t, y

def harmonic_oscillator(t, y):
    """
    System of ODEs for a harmonic oscillator
    y[0] = position
    y[1] = velocity
    """
    return np.array([y[1], -y[0]])

@jit(nopython=True)
def hco_morris_lecar(t, y, Iext0=0.0e-6, Iext1=0.0e-6, IextJ=0.4e-6, Cap=1e-6, phiN=0.000002e3,V_slope=0.001e-3,g_Syn=6.0e-6):
    # constants
    #phiN = 0.000002e3  # seconds^-1 

    V_Ca = 100e-3  # volts
    V_K = -80e-3  # volts
    V_L = -50e-3  # volts
    V_Syn = -30e-3  # volts

    V_1 = 0e-3  # volts
    V_2 = 15e-3  # volts
    V_3 = 0e-3  # volts
    V_4 = 15e-3  # volts
    #V_slope = 0.001e-3  # volts
    V_thresh = 0e-3  # volts

    g_K = 20e-6  # siemens /cm2
    g_Ca = 15e-6  # siemens /cm2
    g_L = 5e-6  # siemens /cm2
    #g_Syn = 6e-6  # siemens / cm2

    V1=y[0]
    V2=y[1]
    N1=y[2]
    N2=y[3]

    # calculate d/dt's
    Minf1 = 0.5 * (1 + np.tanh((V1 - V_1) / V_2))
    Ninf1 = 0.5 * (1 + np.tanh((V1 - V_3) / V_4))

    Lambda1 = phiN * np.cosh((V1 - V_3) / (2 * V_4))
    Sinf1 = 0.5 * (1 + np.tanh((V2 - V_thresh) / V_slope))  # cross coupling

    Minf2 = 0.5 * (1 + np.tanh((V2 - V_1) / V_2))
    Ninf2 = 0.5 * (1 + np.tanh((V2 - V_3) / V_4))

    Lambda2 = phiN * np.cosh((V2 - V_3) / (2 * V_4))
    Sinf2 = 0.5 * (1 + np.tanh((V1 - V_thresh) / V_slope))  # cross coupling

    V1dot = (1 / Cap) * (Iext0+IextJ - g_L * (V1 - V_L) - g_Ca * Minf1 * (V1 - V_Ca) -
                        g_K * N1 * (V1 - V_K) - g_Syn * Sinf1 * (V1 - V_Syn))

    V2dot = (1 / Cap) * (Iext1+IextJ - g_L * (V2 - V_L) - g_Ca * Minf2 * (V2 - V_Ca) -
                        g_K * N2 * (V2 - V_K) - g_Syn * Sinf2 * (V2 - V_Syn))

    N1dot = Lambda1 * (Ninf1 - N1)
    N2dot = Lambda2 * (Ninf2 - N2)

    return np.array([V1dot, V2dot, N1dot, N2dot])

def generate_hco_fast(t, initial_conditions, dt):
    # this is a fast HCO with high external input and higher frequency
    # generated then used as external input for the main HCO
    # it just returns the V1 and V2 values for the main HCO to use
    # the parameters are selected to make frequency high and dynamics sinusoidalish

    def hco_wrapper_const(t, y):
        return hco_morris_lecar(t, y, Iext0=0.1e-6, Iext1=0.0e-6, IextJ=2.2e-6, Cap=25.1e-6, phiN=0.0181, V_slope=16e-3, g_Syn=60e-6)
    
    y = np.zeros((len(t), len(initial_conditions)))
    y[0] = initial_conditions
    
    for i in range(1, len(t)):
        dy = hco_wrapper_const(t[i-1], y[i-1])
        y[i] = y[i-1] + dt * dy
    
    v1_fast = y[:, 0]
    v2_fast = y[:, 1]

    return v1_fast, v2_fast


@jit(nopython=True)
def hco_morris_lecar2(t, y, Iext0=3.05e-6, Iext1=3.05e-6, Cap=1e-6, phiN=0.000002e3,V_slope=15e-3):
#sinusoidal shape

    # constants
    #phiN = 0.000002e3  # seconds^-1 #from fig 11

    V_Ca = 100e-3  # volts  #AppendixA
    V_K = -80e-3  # volts   #AppendixA
    V_L = -50e-3  # volts   #AppendixA
    V_Syn = -80e-3  # volts   #AppendixA

    V_1 = 0e-3  # volts  #AppendixA
    V_2 = 15e-3  # volts  #AppendixA
    V_3 = 0e-3  # volts   #AppendixA
    V_4 = 15e-3  # volts  #AppendixA
    #V_slope = 15e-3  # volts  #from fig 11
    V_thresh = 0e-3  # volts  #from fig 11

    g_K = 20e-6  # siemens /cm2     #AppendixA
    g_Ca = 15e-6  # siemens /cm2   #AppendixA
    g_L = 5e-6  # siemens /cm2     #AppendixA
    g_Syn = 30e-6  # siemens / cm2  #from figure 11

    V1=y[0]
    V2=y[1]
    N1=y[2]
    N2=y[3]

    # calculate d/dt's
    Minf1 = 0.5 * (1 + np.tanh((V1 - V_1) / V_2))
    Ninf1 = 0.5 * (1 + np.tanh((V1 - V_3) / V_4))

    Lambda1 = phiN * np.cosh((V1 - V_3) / (2 * V_4))
    Sinf1 = 0.5 * (1 + np.tanh((V2 - V_thresh) / V_slope))  # cross coupling

    Minf2 = 0.5 * (1 + np.tanh((V2 - V_1) / V_2))
    Ninf2 = 0.5 * (1 + np.tanh((V2 - V_3) / V_4))

    Lambda2 = phiN * np.cosh((V2 - V_3) / (2 * V_4))
    Sinf2 = 0.5 * (1 + np.tanh((V1 - V_thresh) / V_slope))  # cross coupling

    V1dot = (1 / Cap) * (Iext0 - g_L * (V1 - V_L) - g_Ca * Minf1 * (V1 - V_Ca) -
                        g_K * N1 * (V1 - V_K) - g_Syn * Sinf1 * (V1 - V_Syn))

    V2dot = (1 / Cap) * (Iext1 - g_L * (V2 - V_L) - g_Ca * Minf2 * (V2 - V_Ca) -
                        g_K * N2 * (V2 - V_K) - g_Syn * Sinf2 * (V2 - V_Syn))

    N1dot = Lambda1 * (Ninf1 - N1)
    N2dot = Lambda2 * (Ninf2 - N2)

    return np.array([V1dot, V2dot, N1dot, N2dot])

def generate_ou_process(t, dt, mu, sigma, tau):
    """
    Generate an Ornstein-Uhlenbeck process.
    
    Parameters:
    -----------
    t : array
        Time points
    dt : float
        Time step
    mu : float
        Mean of the process
    sigma : float
        Standard deviation of the process
    tau : float
        Time constant of the process
    
    Returns:
    --------
    x : array
        Ornstein-Uhlenbeck process values
    """
    n_steps = len(t)
    x = np.zeros(n_steps)
    x[0] = mu  # Start at mean
    
    # Parameters for the OU process
    theta = 1/tau  # Mean reversion rate
    sigma_dt = sigma * np.sqrt(dt)  # Scaled noise
    
    # Generate Wiener process increments
    dW = np.random.normal(0, 1, n_steps-1) * sigma_dt
    
    # Euler-Maruyama method for OU process
    for i in range(1, n_steps):
        x[i] = x[i-1] + theta * (mu - x[i-1]) * dt + dW[i-1]
    
    return x




def run_interactive_simulation():
    
    # Create the main figure and subplots
    fig = plt.figure(figsize=(15, 10))#, constrained_layout=True)  # Made figure wider
    
    # define overall window layout
    gs_main = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1.5], width_ratios=[1, 1], figure=fig)

    # Define subplots using the main GridSpec
    ax_main = fig.add_subplot(gs_main[0, 0])  # Top-left: main simulation plot
    ax_hist = fig.add_subplot(gs_main[0, 1])  # Top-right: histogram
    ax_recovery = fig.add_subplot(gs_main[1, 0])  # Middle-left: recovery plot

    # Define a nested GridSpec for the sliders within the bottom row of gs_main
    # slider_gs = gs_main[2, :].subgridspec(12, 1)  # Occupies the last row of gs_main
    slider_gs = gs_main[2, :].subgridspec(13, 2, width_ratios=[4, 1], wspace=0.3)  

    # Create slider axes using the separate GridSpec
    slider_labels = [
        "ax_time", "ax_dt", "ax_iext_mu", "ax_iext_sigma", "ax_iext_tau",
        "ax_sin_amp", "ax_sin_freq", "ax_hco_fast_amp", "ax_iextJ", "ax_cap", "ax_phiN",
        "ax_V_slope", "ax_g_Syn"
    ]
    slider_axes = {label: fig.add_subplot(slider_gs[i, 0]) for i, label in enumerate(slider_labels)}
    ax_time, ax_dt, ax_iext_mu, ax_iext_sigma, ax_iext_tau, \
    ax_sin_amp, ax_sin_freq, ax_hco_fast_amp, ax_iextJ, ax_cap, ax_phiN, \
    ax_V_slope, ax_g_Syn = [slider_axes[key] for key in slider_axes]

    # Hide x/y ticks on slider axes
    for ax in slider_axes.values():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)  # Hide the border

    # Create axes for the radio buttons and save button, positioned on the right
    ax_radio = fig.add_subplot(slider_gs[:6, 1])  # Top half of right column for radio buttons
    ax_save = fig.add_subplot(slider_gs[8:10, 1])  # Lower section for save button
    
    # Create save button
    save_button = Button(ax_save, 'Save', color='lightgray', hovercolor='0.975')
    
    # Create directories for saving if they don't exist
    os.makedirs('simulation_data', exist_ok=True)
    os.makedirs('simulation_figures', exist_ok=True)
    
    def save_simulation(event):
        # Get current timestamp for unique filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Collect current parameters
        params = {
            'system': radio.value_selected,
            'total_time': time_slider.val,
            'dt': dt_slider.val,
            'iext0_mean': iext_mu_slider.val,
            'iext0_std': iext_sigma_slider.val,
            'iext0_tau': iext_tau_slider.val,
            'sin_amplitude': sin_amp_slider.val,
            'sin_frequency': sin_freq_slider.val,
            'hco_fast_amp': hco_fast_amp_slider.val,
            'iextJ': iextJ_slider.val,
            'cap': cap_slider.val,
            'phiN': phiN_slider.val,
            'V_slope': V_slope_slider.val,
            'g_Syn': g_Syn_slider.val,
            'timestamp': timestamp,
            'filename': f'simulation_figures/figure_{timestamp}.png'
            
        }
        
        # Save parameters to JSON file
        param_filename = f'simulation_data/params_{timestamp}.json'
        with open(param_filename, 'w') as f:
            json.dump(params, f, indent=4)
        
        # Save figure
        fig_filename = f'simulation_figures/figure_{timestamp}.png'
        fig.savefig(fig_filename, dpi=300, bbox_inches='tight')
        
        print(f"Saved simulation data to {param_filename}")
        print(f"Saved figure to {fig_filename}")
    
    # Register save button callback
    save_button.on_clicked(save_simulation)
    
    # Create sliders
    time_slider = Slider(
        ax=ax_time,
        label='Total Time (s)',
        valmin=100,
        valmax=20000,
        valinit=1000,
        valstep=100
    )
    
    dt_slider = Slider(
        ax=ax_dt,
        label='Time Step',
        valmin=0.01,
        valmax=0.1,
        valinit=0.02,
        valstep=0.01
    )
    
    # Create OU process parameter sliders
    iext_mu_slider = Slider(
        ax=ax_iext_mu,
        label='Iext0 Mean (µA/cm²)',
        valmin=0,
        valmax=1.0,
        valinit=0.4,
        valstep=0.1
    )

    iext_sigma_slider = Slider(
        ax=ax_iext_sigma,
        label='Iext0 Std Dev (µA/cm²)',
        valmin=0,
        valmax=0.5,
        valinit=0.1,
        valstep=0.05
    )

    iext_tau_slider = Slider(
        ax=ax_iext_tau,
        label='Iext0 Time Constant (s)',
        valmin=0.1,
        valmax=10.0,
        valinit=8.2,
        valstep=0.1
    )

    # Create sinusoidal input sliders
    sin_amp_slider = Slider(
        ax=ax_sin_amp,
        label='Sinusoidal Amplitude (µA/cm²)',
        valmin=0,
        valmax=1.0,
        valinit=0,
        valstep=0.1
    )

    sin_freq_slider = Slider(
        ax=ax_sin_freq,
        label='Sinusoidal Frequency (Hz)',
        valmin=0.01,
        valmax=1.0,
        valinit=0,
        valstep=0.01
    )

    # Create input slider for magnitude of input from faster HCO
    hco_fast_amp_slider = Slider(
        ax=ax_hco_fast_amp,
        label='Fast HCO Input Amplitude',
        valmin=0,
        valmax=10.0,
        valinit=4,
        valstep=0.1
    )

    # Create Iext slider with scientific notation
    iextJ_slider = Slider(
        ax=ax_iextJ,
        label='IextJoint (µA/cm²)',
        valmin=0,
        valmax=5.0,
        valinit=0.4,
        valstep=0.1
    )

    # Create Cap slider
    cap_slider = Slider(
        ax=ax_cap,
        label='Cap (µF/cm²)',
        valmin=0.1,
        valmax=40.0,
        valinit=1.1,
        valstep=0.5
    )

    # Create PhiN slider
    phiN_slider = Slider(
        ax=ax_phiN,
        label='phiN (1/msec)',
        valmin=0.0001,
        valmax=0.1,
        valinit=0.002,
        valstep=0.002
    )

    # Create V_slope slider
    V_slope_slider = Slider(
        ax=ax_V_slope,
        label='V_slope (mV)',
        valmin=1,
        valmax=100,
        valinit=16,
        valstep=5
    )
    
    # Create g_Syn slider
    g_Syn_slider = Slider(
        ax=ax_g_Syn,
        label='g_Syn (µS/cm²)',
        valmin=0,
        valmax=100,
        valinit=30,
        valstep=5
    )

    # Reduce font size for all slider labels
    sliders = [time_slider, dt_slider, iext_mu_slider, iext_sigma_slider, iext_tau_slider, 
            sin_amp_slider, sin_freq_slider, hco_fast_amp_slider, iextJ_slider, cap_slider, phiN_slider, 
            V_slope_slider, g_Syn_slider]

    for slider in sliders:
        slider.label.set_size(8)

    # Create radio buttons for system selection
    radio = RadioButtons(
        ax_radio,
        ('HCO', 'Oscillator'),
        active=0
    )
    
    def update(val=None):
        # Clear current plots
        ax_main.clear()
        ax_recovery.clear()
        ax_hist.clear()
        
        # Get current values
        total_time = time_slider.val
        dt = dt_slider.val
        system = radio.value_selected
        t_span = (0, total_time)
        
        if system == 'HCO':
            initial_conditions = [5e-3, -5e-3, 0.7, 0.06]
            t_start = time.time()
            
            # Generate time array
            t = np.arange(t_span[0], t_span[1] + dt, dt)
            
            # Generate OU process for Iext0
            iext0_mu = iext_mu_slider.val * 1e-6  # Convert to A/cm²
            iext0_sigma = iext_sigma_slider.val * 1e-6  # Convert to A/cm²
            iext0_tau = iext_tau_slider.val
            iext0 = generate_ou_process(t, dt, iext0_mu, iext0_sigma, iext0_tau)
            
            # Generate sinusoidal input
            sin_amp = sin_amp_slider.val * 1e-6  # Convert to A/cm²
            sin_freq = sin_freq_slider.val
            sin_input = sin_amp * np.sin(2 * np.pi * sin_freq * t)

            ####### "coupling" via external input which is the voltage difference between the cells from another HCO
            # this "fast HCO" is parameterized with high external input and higher frequency than the main HCO
            hco_fast_amp = hco_fast_amp_slider.val * 1e-6  # Convert to A/cm²
            v1_fast, v2_fast = generate_hco_fast(t, initial_conditions, dt) # same initial conditions as slow/main HCO
            hco_fast = hco_fast_amp * (v1_fast - v2_fast)
            #########

            # Combine OU and HCO inputs and sinusoidal inputs
            iext0_total = iext0 + hco_fast + sin_input

            # Convert other slider values
            iextJ = iextJ_slider.val * 1e-6
            cap = cap_slider.val * 1e-6     
            phiN = phiN_slider.val  
            V_slope = V_slope_slider.val * 1e-3
            g_Syn = g_Syn_slider.val * 1e-6

            # Create wrapper function to pass Iext and Cap
            def hco_wrapper(t, y, i):
                return hco_morris_lecar(t, y, Iext0=iext0_total[i], IextJ=iextJ, Cap=cap, phiN=phiN, V_slope=V_slope, g_Syn=g_Syn)
            
            # Modified Euler simulation to handle time-dependent Iext0
            y = np.zeros((len(t), len(initial_conditions)))
            y[0] = initial_conditions
            
            for i in range(1, len(t)):
                dy = hco_wrapper(t[i-1], y[i-1], i-1)
                y[i] = y[i-1] + dt * dy
            
            t_end = time.time()
            print(f"Simulation time elapsed (s): {t_end-t_start:.3f}")

            ax_main.plot(t, y[:, 0], 'r-', label='V1')
            ax_main.plot(t, y[:, 1], 'b-', label='V2')
            ax_main.plot(t, iext0_total/1e-6, 'g--', label='Total Iext0 (µA/cm²)', alpha=0.5)
            ax_main.plot(t, hco_fast/1e-6, 'm--', label='Faster HCO Input (V1_fast-V2_fast)', alpha=0.5)
            ax_recovery.plot(t, y[:, 2], 'r-', label='N1')
            ax_recovery.plot(t, y[:, 3], 'b-', label='N2')


            # Plot membrane potentials
            ax_main.set_xlabel('Time (s)')
            ax_main.set_ylabel('Voltage (V)')
            ax_main.set_title('Half-Center Oscillator - Membrane Potentials')
            ax_main.legend()
            ax_main.grid(True)

            # Plot recovery variables
            ax_recovery.set_xlabel('Time (s)')
            ax_recovery.set_ylabel('Activation')
            ax_recovery.set_title('Recovery Variables')
            ax_recovery.legend()
            ax_recovery.grid(True)
            ax_recovery.set_visible(True)
            
            # Calculate and plot histogram of positive intervals
            V1 = y[:, 0]
            positive_intervals = []
            current_interval = 0
            
            for v in V1:
                if v > 0:
                    current_interval += 1
                elif current_interval > 0:
                    positive_intervals.append(current_interval * dt)  # Convert to seconds
                    current_interval = 0
            
            # Add the last interval if it's positive
            if current_interval > 0:
                positive_intervals.append(current_interval * dt)
            
            if positive_intervals:
                ax_hist.hist(positive_intervals, bins=30, edgecolor='black')
                ax_hist.set_xlabel('Duration (s)')
                ax_hist.set_ylabel('Count')
                ax_hist.set_title('Histogram of Positive V1 Intervals')
                ax_hist.grid(True)
            else:
                ax_hist.text(0.5, 0.5, 'No positive intervals found', 
                           horizontalalignment='center',
                           verticalalignment='center',
                           transform=ax_hist.transAxes)
            
            # Show all sliders
            ax_iext_mu.set_visible(True)
            ax_iext_sigma.set_visible(True)
            ax_iext_tau.set_visible(True)
            ax_iextJ.set_visible(True)
            ax_cap.set_visible(True)
            ax_phiN.set_visible(True)
            ax_V_slope.set_visible(True)
            ax_g_Syn.set_visible(True)

        elif system == 'HCO2':
            pass

        else:  # Harmonic oscillator
            initial_conditions = [1.0, 0.0]
            t, y = euler_simulator(harmonic_oscillator, initial_conditions, t_span, dt)
            
            ax_main.plot(t, y[:, 0], label='Position')
            ax_main.plot(t, y[:, 1], label='Velocity')
            ax_main.set_xlabel('Time (s)')
            ax_main.set_ylabel('State Variables')
            ax_main.set_title('Harmonic Oscillator Simulation')
            ax_main.legend()
            ax_main.grid(True)
            
            # Hide all parameter sliders for harmonic oscillator
            ax_iext_mu.set_visible(False)
            ax_iext_sigma.set_visible(False)
            ax_iext_tau.set_visible(False)
            ax_iextJ.set_visible(False)
            ax_cap.set_visible(False)
            ax_phiN.set_visible(False)
            ax_V_slope.set_visible(False)
            ax_g_Syn.set_visible(False)
            ax_recovery.set_visible(False)
            ax_hist.set_visible(False)
        
        # get canvas to respond if events emitted
        fig.canvas.draw_idle()
    
    # Register update function with the widgets
    time_slider.on_changed(update)
    dt_slider.on_changed(update)
    iext_mu_slider.on_changed(update)
    iext_sigma_slider.on_changed(update)
    iext_tau_slider.on_changed(update)
    sin_amp_slider.on_changed(update)
    sin_freq_slider.on_changed(update)
    hco_fast_amp_slider.on_changed(update)
    iextJ_slider.on_changed(update)
    cap_slider.on_changed(update)
    phiN_slider.on_changed(update)
    V_slope_slider.on_changed(update)
    g_Syn_slider.on_changed(update)
    radio.on_clicked(update)
    
    # Initial plot
    update()
    
    plt.show()

if __name__ == "__main__":
    run_interactive_simulation() 