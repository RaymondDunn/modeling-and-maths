import numpy as np
import time as time
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
from numba import jit
import json
import os
from datetime import datetime
import matplotlib.gridspec as gridspec

@jit(nopython=True)
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


# function to arrange axes of the figure
def init_fig_axes(slider_ax_names):

    # Create the main figure and subplots
    fig = plt.figure(figsize=(15, 10))#, constrained_layout=True)  # Made figure wider
    
    # define overall window layout
    gs_main = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1.5], width_ratios=[1, 1], figure=fig)

    # Define subplots using the main GridSpec
    ax_main = fig.add_subplot(gs_main[0, 0])  # Top-left: main simulation plot
    ax_hist = fig.add_subplot(gs_main[0, 1])  # Top-right: histogram
    ax_recovery = fig.add_subplot(gs_main[1, 0])  # Middle-left: recovery plot

    # Define a nested GridSpec for the sliders within the bottom row of gs_main
    slider_gs = gs_main[2, :].subgridspec(len(slider_ax_names), 2, width_ratios=[4, 1], wspace=0.3)
    slider_axes_dict = {label: fig.add_subplot(slider_gs[i, 0]) for i, label in enumerate(slider_ax_names)}

    # Create axes for the radio buttons and save button, positioned on the right
    ax_radio = fig.add_subplot(slider_gs[:6, 1])  # Top half of right column for radio buttons
    ax_save = fig.add_subplot(slider_gs[8:10, 1])  # Lower section for save button

    ## do some formatting
    # Hide x/y ticks on slider axes
    for ax in slider_axes_dict.values():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)  # Hide the border

    # pack objects into dicts
    plot_axes_dict = {
        'ax_main': ax_main,
        'ax_hist': ax_hist,
        'ax_recovery': ax_recovery
    }
    gridspec_dict = {
        'gs_main': gs_main,
        'slider_gs': slider_gs
    }
    radio_axes_dict = {
        'ax_radio': ax_radio,
        'ax_save': ax_save
    }

    return fig, gridspec_dict, plot_axes_dict, slider_axes_dict, radio_axes_dict



class Simulation:

    # make everything a class attribute
    def __init__(self, fig, **kwargs):
        self.fig = fig
        self.total_time = None
        self.dt = None
        self.t_span = None
        self.add_attributes(**kwargs)

    def set_simulation_scope(self, total_time, dt, t_span):
        self.total_time = total_time
        self.dt = dt
        self.t_span = t_span
        
    def add_attributes(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    # function to create gui elements
    def init_gui_elements(self, **kwargs):

        # Create directories for saving if they don't exist
        os.makedirs('simulation_data', exist_ok=True)
        os.makedirs('simulation_figures', exist_ok=True)
        
        # Create sliders
        self.time_slider = Slider(
            ax=self.ax_time,
            label='Total Time (s)',
            valmin=100,
            valmax=20000,
            valinit=1000,
            valstep=100
        )
        
        self.dt_slider = Slider(
            ax=self.ax_dt,
            label='Time Step',
            valmin=0.01,
            valmax=0.1,
            valinit=0.02,
            valstep=0.01
        )
        
        # Create OU process parameter sliders
        self.iext_mu_slider = Slider(
            ax=self.ax_iext_mu,
            label='Iext0 Mean (µA/cm²)',
            valmin=0,
            valmax=1.0,
            valinit=0.4,
            valstep=0.1
        )

        self.iext_sigma_slider = Slider(
            ax=self.ax_iext_sigma,
            label='Iext0 Std Dev (µA/cm²)',
            valmin=0,
            valmax=0.5,
            valinit=0.1,
            valstep=0.05
        )

        self.iext_tau_slider = Slider(
            ax=self.ax_iext_tau,
            label='Iext0 Time Constant (s)',
            valmin=0.1,
            valmax=10.0,
            valinit=8.2,
            valstep=0.1
        )

        # Create sinusoidal input sliders
        self.sin_amp_slider = Slider(
            ax=self.ax_sin_amp,
            label='Sinusoidal Amplitude (µA/cm²)',
            valmin=0,
            valmax=1.0,
            valinit=0.3,
            valstep=0.1
        )

        self.sin_freq_slider = Slider(
            ax=self.ax_sin_freq,
            label='Sinusoidal Frequency (Hz)',
            valmin=0.01,
            valmax=1.0,
            valinit=0.02,
            valstep=0.01
        )

        # Create Iext slider with scientific notation
        self.iextJ_slider = Slider(
            ax=self.ax_iextJ,
            label='IextJoint (µA/cm²)',
            valmin=0,
            valmax=5.0,
            valinit=0.4,
            valstep=0.1
        )

        # Create Cap slider
        self.cap_slider = Slider(
            ax=self.ax_cap,
            label='Cap (µF/cm²)',
            valmin=0.1,
            valmax=40.0,
            valinit=1.1,
            valstep=0.5
        )

        # Create PhiN slider
        self.phiN_slider = Slider(
            ax=self.ax_phiN,
            label='phiN (1/msec)',
            valmin=0.0001,
            valmax=0.1,
            valinit=0.002,
            valstep=0.002
        )

        # Create V_slope slider
        self.V_slope_slider = Slider(
            ax=self.ax_V_slope,
            label='V_slope (mV)',
            valmin=1,
            valmax=100,
            valinit=16,
            valstep=5
        )
        
        # Create g_Syn slider
        self.g_Syn_slider = Slider(
            ax=self.ax_g_Syn,
            label='g_Syn (µS/cm²)',
            valmin=0,
            valmax=100,
            valinit=30,
            valstep=5
        )

        ##############################################################
        # as we expand in complexity we want more sliders
        self.fast_hco_amp_slider = Slider(
            ax=self.ax_fast_hco_amp,
            label='Fast HCO input amplitude (µA/cm²)',
            valmin=0,
            valmax=10.0,
            valinit=0.3,
            valstep=0.1
        )

        # helpful to have a list of sliders
        self.slider_list = [self.time_slider, self.dt_slider, self.iext_mu_slider, self.iext_sigma_slider, self.iext_tau_slider, 
                self.iextJ_slider, self.cap_slider, self.phiN_slider, self.V_slope_slider, self.g_Syn_slider, self.sin_amp_slider, self.sin_freq_slider, self.fast_hco_amp_slider]
        
        for slider in self.slider_list:
            slider.label.set_size(10)

        # Create radio buttons for system selection
        self.radio = RadioButtons(
            self.ax_radio,
            ('HCO', 'HCO sine', 'Two HCOs'),
            active=0
        )

        # create buttons
        self.save_button = Button(self.ax_save, 'Save', color='lightgray', hovercolor='0.975')

        # map callbacks
        self.map_callback_to_elements()

    
    def map_callback_to_elements(self):

        ## callbacks
        def save_simulation(event):

            # Get current timestamp for unique filenames
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Collect current parameters
            params = {
                'system': self.radio.value_selected,
                'timestamp': timestamp
            }
            for slider_ax_name in self.slider_ax_names:
                params[slider_ax_name] = getattr(self, slider_ax_name).val
            
            # Save parameters to JSON file
            param_filename = f'simulation_data/params_{timestamp}.json'
            with open(param_filename, 'w') as f:
                json.dump(params, f, indent=4)
            
            # Save figure
            fig_filename = f'simulation_figures/figure_{timestamp}.png'
            self.fig.savefig(fig_filename, dpi=300, bbox_inches='tight')
            
            print(f"Saved simulation data to {param_filename}")
            print(f"Saved figure to {fig_filename}")
        
        # Connect save button to save_simulation function
        self.save_button.on_clicked(save_simulation)

        # Register update function with the widgets
        for slider in self.slider_list:
            slider.on_changed(self.update)

        # Register update function with radio buttons
        self.radio.on_clicked(self.update)


    def update(self, val=None):
        
        # Clear current plots
        self.ax_main.clear()
        self.ax_recovery.clear()
        self.ax_hist.clear()
        
        # Get current values
        total_time = self.time_slider.val
        dt = self.dt_slider.val
        system = self.radio.value_selected
        t_span = (0, total_time)
        self.set_simulation_scope(total_time, dt, t_span)
        
        # configure layout of simulation
        if system == 'HCO':
            self.configure_simulation_HCO()
        elif system == 'HCO2':
            self.configure_simulation_HCO2()
        elif system == 'Two HCOs':
            self.configure_simulation_two_HCOs()
        else:  
            self.configure_simulation_harmonic_OSC()
        self.fig.canvas.draw_idle()


    def turn_off_all_sliders(self):
        for slider_ax_name in self.slider_ax_names:
            getattr(self, slider_ax_name).set_visible(False)

    def turn_on_all_sliders(self):
        for slider_ax_name in self.slider_ax_names:
            getattr(self, slider_ax_name).set_visible(True)


    # configure gui elements according to the desired layout
    def configure_simulation_HCO(self):
            
        initial_conditions = [5e-3, -5e-3, 0.7, 0.06]
        t_start = time.time()
        
        # Generate time array
        t = np.arange(self.t_span[0], self.t_span[1] + self.dt, self.dt)
        
        # Generate OU process for Iext0
        iext0_mu = self.iext_mu_slider.val * 1e-6  # Convert to A/cm²
        iext0_sigma = self.iext_sigma_slider.val * 1e-6  # Convert to A/cm²
        iext0_tau = self.iext_tau_slider.val
        iext0 = generate_ou_process(t, self.dt, iext0_mu, iext0_sigma, iext0_tau)
        
        # Generate sinusoidal input
        sin_amp = self.sin_amp_slider.val * 1e-6  # Convert to A/cm²
        sin_freq = self.sin_freq_slider.val
        sin_input = sin_amp * np.sin(2 * np.pi * sin_freq * t)
        
        # Combine OU and sinusoidal inputs
        iext0_total = iext0 + sin_input
        
        # Convert other slider values
        iextJ = self.iextJ_slider.val * 1e-6
        cap = self.cap_slider.val * 1e-6     
        phiN = self.phiN_slider.val  
        V_slope = self.V_slope_slider.val * 1e-3
        g_Syn = self.g_Syn_slider.val * 1e-6

        # Create wrapper function to pass Iext and Cap
        def hco_wrapper(t, y, i):
            return hco_morris_lecar(t, y, Iext0=iext0_total[i], IextJ=iextJ, Cap=cap, phiN=phiN, V_slope=V_slope, g_Syn=g_Syn)
        
        # Modified Euler simulation to handle time-dependent Iext0
        y = np.zeros((len(t), len(initial_conditions)))
        y[0] = initial_conditions
        
        for i in range(1, len(t)):
            dy = hco_wrapper(t[i-1], y[i-1], i-1)
            y[i] = y[i-1] + self.dt * dy
        
        t_end = time.time()
        print(f"Simulation time elapsed (s): {t_end-t_start:.3f}")
        
        # Plot membrane potentials
        self.ax_main.plot(t, y[:, 0], 'r-', label='V1')
        self.ax_main.plot(t, y[:, 1], 'b-', label='V2')
        self.ax_main.plot(t, iext0_total/1e-6, 'g--', label='Total Iext0 (µA/cm²)', alpha=0.5)
        self.ax_main.plot(t, sin_input/1e-6, 'm--', label='Sinusoidal Input (µA/cm²)', alpha=0.5)
        self.ax_main.set_xlabel('Time (s)')
        self.ax_main.set_ylabel('Voltage (V)')
        self.ax_main.set_title('Half-Center Oscillator - Membrane Potentials')
        self.ax_main.legend()
        self.ax_main.grid(True)
        
        # Plot recovery variables
        self.ax_recovery.plot(t, y[:, 2], 'r-', label='N1')
        self.ax_recovery.plot(t, y[:, 3], 'b-', label='N2')
        self.ax_recovery.set_xlabel('Time (s)')
        self.ax_recovery.set_ylabel('Activation')
        self.ax_recovery.set_title('Recovery Variables')
        self.ax_recovery.legend()
        self.ax_recovery.grid(True)
        self.ax_recovery.set_visible(True)
        
        # Calculate and plot histogram of positive intervals
        V1 = y[:, 0]
        positive_intervals = []
        current_interval = 0
        
        for v in V1:
            if v > 0:
                current_interval += 1
            elif current_interval > 0:
                positive_intervals.append(current_interval * self.dt)  # Convert to seconds
                current_interval = 0
        
        # Add the last interval if it's positive
        if current_interval > 0:
            positive_intervals.append(current_interval * self.dt)
        
        if positive_intervals:
            self.ax_hist.hist(positive_intervals, bins=30, edgecolor='black')
            self.ax_hist.set_xlabel('Duration (s)')
            self.ax_hist.set_ylabel('Count')
            self.ax_hist.set_title('Histogram of Positive V1 Intervals')
            self.ax_hist.grid(True)
        else:
            self.ax_hist.text(0.5, 0.5, 'No positive intervals found', 
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=self.ax_hist.transAxes)
        
        # Show appropriate sliders
        self.turn_off_all_sliders()
        self.ax_iext_mu.set_visible(True)
        self.ax_iext_sigma.set_visible(True)
        self.ax_iext_tau.set_visible(True)
        self.ax_iextJ.set_visible(True)
        self.ax_cap.set_visible(True)
        self.ax_phiN.set_visible(True)
        self.ax_V_slope.set_visible(True)
        self.ax_g_Syn.set_visible(True)


    def configure_simulation_HCO2(self):
        pass
        # initial_conditions = [5e-3, -5e-3, 0.7, 0.06]
        # t_start = time.time()
        
        # # Generate OU process for Iext0
        # t = np.arange(self.t_span[0], self.t_span[1] + self.dt, self.dt)
        # iext0_mu = self.iext_mu_slider.val * 1e-6  # Convert to A/cm²
        # iext0_sigma = self.iext_sigma_slider.val * 1e-6  # Convert to A/cm²
        # iext0_tau = self.iext_tau_slider.val
        # iext0 = generate_ou_process(t, self.dt, iext0_mu, iext0_sigma, iext0_tau)
        
        # # Convert other slider values
        # cap = self.cap_slider.val * 1e-6
        # phiN = self.phiN_slider.val    

        # # Create wrapper function to pass Iext and Cap
        # def hco_wrapper(t, y, i):
        #     return hco_morris_lecar2(t, y, Iext0=iext0[i], Cap=cap, phiN=phiN)
        
        # # Modified Euler simulation to handle time-dependent Iext0
        # y = np.zeros((len(t), len(initial_conditions)))
        # y[0] = initial_conditions
        
        # for i in range(1, len(t)):
        #     dy = hco_wrapper(t[i-1], y[i-1], i-1)
        #     y[i] = y[i-1] + self.dt * dy
        
        # t_end = time.time()
        # print(f"Simulation time elapsed (s): {t_end-t_start:.3f}")
        
        # # Plot membrane potentials
        # self.ax_main.plot(t, y[:, 0], 'r-', label='V1')
        # self.ax_main.plot(t, y[:, 1], 'b-', label='V2')
        # self.ax_main.plot(t, iext0/1e-6, 'g--', label='Iext0 (µA/cm²)', alpha=0.5)
        # self.ax_main.set_xlabel('Time (s)')
        # self.ax_main.set_ylabel('Voltage (V)')
        # self.ax_main.set_title('Half-Center Oscillator - Membrane Potentials')
        # self.ax_main.legend()
        # self.ax_main.grid(True)
        
        # # Plot recovery variables
        # self.ax_recovery.plot(t, y[:, 2], 'r-', label='N1')
        # self.ax_recovery.plot(t, y[:, 3], 'b-', label='N2')
        # self.ax_recovery.set_xlabel('Time (s)')
        # self.ax_recovery.set_ylabel('Activation')
        # self.ax_recovery.set_title('Recovery Variables')
        # self.ax_recovery.legend()
        # self.ax_recovery.grid(True)
        # self.ax_recovery.set_visible(True)
        
        # # Calculate and plot histogram of positive intervals
        # V1 = y[:, 0]
        # positive_intervals = []
        # current_interval = 0
        
        # for v in V1:
        #     if v > 0:
        #         current_interval += 1
        #     elif current_interval > 0:
        #         positive_intervals.append(current_interval * self.dt)  # Convert to seconds
        #         current_interval = 0
        
        # # Add the last interval if it's positive
        # if current_interval > 0:
        #     positive_intervals.append(current_interval * self.dt)
        
        # if positive_intervals:
        #     self.ax_hist.hist(positive_intervals, bins=30, edgecolor='black')
        #     self.ax_hist.set_xlabel('Duration (s)')
        #     self.ax_hist.set_ylabel('Count')
        #     self.ax_hist.set_title('Histogram of Positive V1 Intervals')
        #     self.ax_hist.grid(True)
        # else:
        #     self.ax_hist.text(0.5, 0.5, 'No positive intervals found', 
        #                 horizontalalignment='center',
        #                 verticalalignment='center',
        #                 transform=self.ax_hist.transAxes)
        
        # # Show all sliders
        # self.ax_iext_mu.set_visible(True)
        # self.ax_iext_sigma.set_visible(True)
        # self.ax_iext_tau.set_visible(True)
        # self.ax_iextJ.set_visible(True)
        # self.ax_cap.set_visible(True)
        # self.ax_phiN.set_visible(True)
        # self.ax_V_slope.set_visible(True)
        # self.ax_g_Syn.set_visible(True)


    def configure_simulation_harmonic_OSC(self):

        initial_conditions = [1.0, 0.0]
        t, y = euler_simulator(harmonic_oscillator, initial_conditions, self.t_span, self.dt)
        
        self.ax_main.plot(t, y[:, 0], label='Position')
        self.ax_main.plot(t, y[:, 1], label='Velocity')
        self.ax_main.set_xlabel('Time (s)')
        self.ax_main.set_ylabel('State Variables')
        self.ax_main.set_title('Harmonic Oscillator Simulation')
        self.ax_main.legend()
        self.ax_main.grid(True)
        
        # Hide all parameter sliders for harmonic oscillator
        self.turn_off_all_sliders()


    def configure_simulation_HCO_sine(self):

        # run sub function
        self.configure_simulation_HCO()

        # add a few more
        self.ax_sin_amp.set_visible(True)
        self.ax_sin_freq.set_visible(True)


    def configure_simulation_two_HCOs(self):

        # run sub function
        self.configure_simulation_HCO_sine()
        
        # turn on another hco slider
        self.ax_fast_hco_amp.set_visible(True)


# 
def run_interactive_simulation():

    # Create list of sliders we want
    slider_ax_names = [
        "ax_time", "ax_dt", "ax_iext_mu", "ax_iext_sigma", "ax_iext_tau",
        "ax_iextJ", "ax_cap", "ax_phiN",
        "ax_V_slope", "ax_g_Syn", "ax_sin_amp", "ax_sin_freq", "ax_fast_hco_amp"
    ]

    # set up the figure
    fig, gridspec_dict, plot_axes_dict, slider_axes_dict, radio_axes_dict = init_fig_axes(slider_ax_names=slider_ax_names)

    # create object and put these axes into it
    sim = Simulation(fig, **plot_axes_dict, **radio_axes_dict, **slider_axes_dict, **gridspec_dict)

    # add these axis names to the object
    sim.slider_ax_names = slider_ax_names

    # create gui elements (need access to axes)
    sim.init_gui_elements(slider_ax_names=slider_ax_names)

    # Initial plot
    sim.update()
    
    # show fig
    plt.show()

if __name__ == "__main__":
    run_interactive_simulation() 