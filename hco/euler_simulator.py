import numpy as np
import time as time
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from numba import jit


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

# @jit(nopython=True)
def hco_morris_lecar(t, y, Iext0=0.4e-6, Iext1=0.0, Cap=1e-6):
    # constants
    phiN = 0.000002e3  # seconds^-1 

    V_Ca = 100e-3  # volts
    V_K = -80e-3  # volts
    V_L = -50e-3  # volts
    V_Syn = -30e-3  # volts

    V_1 = 0e-3  # volts
    V_2 = 15e-3  # volts
    V_3 = 0e-3  # volts
    V_4 = 15e-3  # volts
    V_slope = 0.001e-3  # volts
    V_thresh = 0e-3  # volts

    g_K = 20e-6  # siemens /cm2
    g_Ca = 15e-6  # siemens /cm2
    g_L = 5e-6  # siemens /cm2
    g_Syn = 6e-6  # siemens / cm2

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

def run_interactive_simulation():
    # Create the main figure and subplots
    fig = plt.figure(figsize=(12, 10))  # Made figure taller
    
    # Add subplots for the simulation
    ax_main = plt.subplot(211)
    ax_recovery = plt.subplot(212)
    
    # Adjust layout to make room for sliders and radio buttons
    plt.subplots_adjust(left=0.15, bottom=0.35)  # Increased bottom margin further
    
    # Create axes for the sliders - moved lower
    ax_time = plt.axes([0.15, 0.20, 0.65, 0.03])  # [left, bottom, width, height]
    ax_dt = plt.axes([0.15, 0.15, 0.65, 0.03])
    ax_iext = plt.axes([0.15, 0.10, 0.65, 0.03])
    ax_cap = plt.axes([0.15, 0.05, 0.65, 0.03])  # New slider for Cap
    
    # Create axes for the radio buttons - moved to the right side
    ax_radio = plt.axes([0.85, 0.08, 0.1, 0.1])
    
    # Create sliders
    time_slider = Slider(
        ax=ax_time,
        label='Total Time (s)',
        valmin=100,
        valmax=2000,
        valinit=1000,
        valstep=100
    )
    
    dt_slider = Slider(
        ax=ax_dt,
        label='Time Step',
        valmin=0.01,
        valmax=0.1,
        valinit=0.05,
        valstep=0.01
    )
    
    # Create Iext slider with scientific notation
    iext_slider = Slider(
        ax=ax_iext,
        label='Iext0 (µA/cm²)',
        valmin=0,
        valmax=1.0,
        valinit=0.4,
        valstep=0.1
    )
    
    # Create Cap slider
    cap_slider = Slider(
        ax=ax_cap,
        label='Cap (µF/cm²)',
        valmin=0.1,
        valmax=2.0,
        valinit=1.0,
        valstep=0.1
    )
    
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
        
        # Get current values
        total_time = time_slider.val
        dt = dt_slider.val
        system = radio.value_selected
        t_span = (0, total_time)
        
        if system == 'HCO':
            initial_conditions = [5e-3, -5e-3, 0.7, 0.06]
            t_start = time.time()
            # Convert Iext slider value from µA/cm² to A/cm²
            iext0 = iext_slider.val * 1e-6
            # Convert Cap slider value from µF/cm² to F/cm²
            cap = cap_slider.val * 1e-6
            
            # Create wrapper function to pass Iext and Cap
            def hco_wrapper(t, y):
                return hco_morris_lecar(t, y, Iext0=iext0, Cap=cap)
            
            t, y = euler_simulator(hco_wrapper, initial_conditions, t_span, dt)
            t_end = time.time()
            print(f"Simulation time elapsed (s): {t_end-t_start:.3f}")
            
            # Plot membrane potentials
            ax_main.plot(t, y[:, 0], 'r-', label='V1')
            ax_main.plot(t, y[:, 1], 'b-', label='V2')
            ax_main.set_xlabel('Time (s)')
            ax_main.set_ylabel('Voltage (V)')
            ax_main.set_title('Half-Center Oscillator - Membrane Potentials')
            ax_main.legend()
            ax_main.grid(True)
            
            # Plot recovery variables
            ax_recovery.plot(t, y[:, 2], 'r-', label='N1')
            ax_recovery.plot(t, y[:, 3], 'b-', label='N2')
            ax_recovery.set_xlabel('Time (s)')
            ax_recovery.set_ylabel('Activation')
            ax_recovery.set_title('Recovery Variables')
            ax_recovery.legend()
            ax_recovery.grid(True)
            ax_recovery.set_visible(True)
            
            # Show Iext and Cap sliders
            ax_iext.set_visible(True)
            ax_cap.set_visible(True)
            
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
            
            # Hide recovery plot and parameter sliders for harmonic oscillator
            ax_recovery.set_visible(False)
            ax_iext.set_visible(False)
            ax_cap.set_visible(False)
        
        # Adjust the layout to prevent overlap
        if system == 'HCO':
            plt.subplots_adjust(hspace=0.3, bottom=0.35)
        else:
            plt.subplots_adjust(bottom=0.35)
            
        fig.canvas.draw_idle()
    
    # Register update function with the widgets
    time_slider.on_changed(update)
    dt_slider.on_changed(update)
    iext_slider.on_changed(update)
    cap_slider.on_changed(update)
    radio.on_clicked(update)
    
    # Initial plot
    update()
    
    plt.show()

if __name__ == "__main__":
    run_interactive_simulation() 