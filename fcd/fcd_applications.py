import mode_fitting
import numpy as np
from dateutil.relativedelta import relativedelta
import utility
import matplotlib.pyplot as plt
import utility_guesses
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.collections import LineCollection
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.collections import LineCollection
from scipy.interpolate import interp1d
testing_mode=1
def show_combined_plot_velocity(x_data, speed_data,y_fit_modes,derivatives_modes):
    fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(7, 6), sharex=True, constrained_layout=True)
    
    ax1.plot(x_data, speed_data, color='gray', alpha=1.0, label='Raw Velocity')
    ax1.set_ylabel('Speed ($km/h$)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Raw Velocity', fontsize=12)
    

    ax2.plot(x_data, y_fit_modes[0], color='blue', linewidth=2, label='SCO Fit')
    ax2.set_ylabel('Speed ($km/h$)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('SCO Optimized Velocity Fit', fontsize=12)
    

    mode = 0  
    y_raw = derivatives_modes[mode]
    
    f = interp1d(x_data, y_raw, kind='cubic')
    x_smooth = np.linspace(x_data.min(), x_data.max(), len(x_data) * 15)
    y_smooth = f(x_smooth)
    
    colors = ["#00008B", "#777777", "#ADFF2F"] 
    my_cmap = LinearSegmentedColormap.from_list("fcd_accel", colors)
    abs_max = max(abs(y_smooth.min()), abs(y_smooth.max())) + 1e-12
    norm = TwoSlopeNorm(vcenter=0, vmin=-abs_max, vmax=abs_max)
    
    points = np.array([x_smooth, y_smooth]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    lc = LineCollection(segments, cmap=my_cmap, norm=norm, linewidth=2.5)
    lc.set_array(y_smooth[:-1])
    ax3.add_collection(lc)
    
    ax3.set_xlim(x_smooth.min(), x_smooth.max())
    ax3.set_ylim(y_smooth.min() * 1.1, y_smooth.max() * 1.1)
    ax3.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax3.set_ylabel('Accel ($km/h^2$)', fontsize=12)
    ax3.set_xlabel('Time ($s$)', fontsize=12)
    ax3.set_title('SCO: Derived Acceleration Fit', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.set_ylim(-2.5, 2.5) 

    fig.align_ylabels([ax1, ax2, ax3])
    plt.savefig("full_plot.png", dpi=300, bbox_inches='tight')
    plt.show()
def show_combined_plot_EEG(x_data, speed_data,y_fit_modes,integrals_modes):
    fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(7.3, 6.3), sharex=True, constrained_layout=True)
    
    ax1.plot(x_data, speed_data, color='gray', alpha=1.0, label='Raw Voltage')
    ax1.set_ylabel('Voltage ($\mu$V)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Raw EEG Data', fontsize=12)
    

    ax2.plot(x_data, y_fit_modes[0], color='blue', linewidth=2, label='SCO Fit')
    ax2.set_ylabel('Voltage ($\mu$V)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('SCO Optimized EEG Fit', fontsize=12)
    
    ax3.plot(x_data, integrals_modes[0], color='green', linewidth=2, label='SCO Fit')
    ax3.set_ylabel('Integrated Voltage ($\mu V \cdot s$)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('SCO Integrated EEG Fit', fontsize=12)

    fig.align_ylabels([ax1, ax2, ax3])
    plt.savefig("full_plot_EEG.png", dpi=300, bbox_inches='tight')
    plt.show()
if testing_mode==0:
    speed_data = np.loadtxt('test_datasets/other_tests/RAW_GPS.txt', usecols=1)
    x_data = np.loadtxt('test_datasets/other_tests/RAW_GPS.txt', usecols=0)
    ch_points=[0, 100,180,200,220,240,260,300,350, 400, 480, 525, 624]
    settings_args={'non_uniform': True, 'changepoints_non_uniform': ch_points}
    mode_fitting_runner=mode_fitting.FCD(x_dataset=x_data,y_dataset=speed_data,model=utility.model_cubic,initial_guesses_function=utility_guesses.initial_guess_cubic,settings_args=settings_args,parallel=True, verbose=1)
    optimized_params=mode_fitting_runner.run()
    y_fit_modes=mode_fitting_runner.calculate_y_fit_modes()
    mode_fitting_runner.print_fitted_functions()
    derivatives_modes=mode_fitting_runner.calculate_derivatives(order=1, print_derivative_formulas=True)
    
    show_combined_plot_velocity(x_data, speed_data,y_fit_modes,derivatives_modes)

    integrals_modes=mode_fitting_runner.calculate_integrals(order=1, print_integral_formulas=False)
    cols=1
    rows=1
    fig,axes=plt.subplots(rows,cols, figsize=(5*rows, 4*cols))
    
    axes_flat=axes
    
    for mode in range(len(integrals_modes)):
        ax = axes_flat
        
        ax.plot(x_data, integrals_modes[mode], color='green', linewidth=2)
        
        changepoint_indices = mode_fitting_runner.all_changepoints[mode]
        for cp_idx in changepoint_indices:
            valid_idx = min(cp_idx, len(x_data) - 1)
            ax.axvline(x=x_data[valid_idx], color='darkslategray', 
                       linestyle='--', linewidth=1.2, alpha=0.6)
        
        ax.set_title(f'Mode {mode+1} Integral', fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.7)
    fig.supxlabel('Time ($t$)', fontsize=14)
    fig.supylabel('Path ($s$)', fontsize=14,x=0.045)

    fig.tight_layout(rect=[0.03, 0.03, 1, 1])
    plt.show()

elif testing_mode==1:
    eeg_data_full = np.load("egg_c3_signal.npy")
    x_data_full=np.load('egg_c3_time.npy')
    y=eeg_data_full[:1000]
    x=x_data_full[:1000]
    
    ch_points=[0, 166, 250,340,420,475,520, 560, 650, 833, 880,920,1000]
    settings_args={'non_uniform': True, 'changepoints_non_uniform': ch_points}
    mode_fitting_runner = mode_fitting.FCD(
        x_dataset=x, y_dataset=y,
        model=utility.model_sin5,
        initial_guesses_function=utility_guesses.initial_guess_sin5,
        parallel=True,
        settings_args=settings_args,
        verbose=1
    )
    mode_fitting_runner.run()
    y_fit_modes=mode_fitting_runner.calculate_y_fit_modes()
    mode_fitting_runner.print_fitted_functions()
    
    integrals_modes=mode_fitting_runner.calculate_integrals(order=1, print_integral_formulas=True)
    show_combined_plot_EEG(x,y,y_fit_modes,integrals_modes)
    cols=2
    rows=3
    fig,axes=plt.subplots(rows,cols, figsize=(5*rows, 4*cols))
    
    axes_flat=np.array([axes]).flatten()
    
    for mode in range(len(integrals_modes)):
        ax = axes_flat[mode]
        
        ax.plot(x, integrals_modes[mode], color='blue', linewidth=2)
        
        changepoint_indices = mode_fitting_runner.all_changepoints[mode]
        for cp_idx in changepoint_indices:
            valid_idx = min(cp_idx, len(x) - 1)
            ax.axvline(x=x[valid_idx], color='darkslategray', 
                       linestyle='--', linewidth=1.2, alpha=0.6)
        
        ax.set_title(f'Mode {mode+1} Integral', fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.7)
    fig.supxlabel('Time ($s$)', fontsize=14)
    fig.supylabel('Integrated Voltage ($\mu V \cdot s$)', fontsize=14,x=0.045)

    fig.tight_layout(rect=[0.03, 0.03, 1, 1])
    plt.show()

    #a1,a0,b0,D,c1,c0
    frequencies_list = [np.array(mode)[:, 2] for mode in mode_fitting_runner.fitted_parameters_modes]
    amplitudes_list  = [np.array(mode)[:, 1] for mode in mode_fitting_runner.fitted_parameters_modes]
    
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#1f77b4', '#ff7f0e', '#8c564b']
    
    t_start = x[0]
    t_end = x[-1]
    
    amplitude_indices = [0, 2, 4, 5]
    
    for i in range(len(frequencies_list)):
        num_pts = len(frequencies_list[i])
        
        if num_pts == 1:
            ax1.hlines(frequencies_list[i][0], t_start, t_end, color=colors[i], 
                       label=f'Mode {i+1} (Global)', linewidth=2)
        else:
            mode_time = np.linspace(t_start, t_end, num_pts)
            ax1.plot(mode_time, frequencies_list[i], label=f'Mode {i+1}', 
                     color=colors[i], linewidth=1.5)
    
    ax1.set_ylabel('Frequency ($b_0$)', fontsize=12)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', ncol=3, fontsize=9)
    fig1.tight_layout()
    plt.savefig('frequency_plot.png', dpi=300)
    plt.show()
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    
    for i in range(len(frequencies_list)):
        num_pts = len(frequencies_list[i])
    
        if i in amplitude_indices:
            if num_pts == 1:
                ax2.hlines(np.abs(amplitudes_list[i][0]), t_start, t_end, color=colors[i], 
                           linewidth=2, linestyle='--', label=f'Mode {i+1} (Global)')
            else:
                mode_time = np.linspace(t_start, t_end, num_pts)
                ax2.plot(mode_time, np.abs(amplitudes_list[i]), color=colors[i], 
                         linewidth=1.5, linestyle='--', label=f'Mode {i+1}')
    
    ax2.set_ylabel('Amplitude ($a_0$)', fontsize=12)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)

    ax2.legend(loc='upper right', ncol=2, fontsize=9)
    fig2.tight_layout()
    plt.savefig('amplitude_plot.png', dpi=300)
    plt.show()
    
    