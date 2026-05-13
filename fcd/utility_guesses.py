import numpy as np
from functools import partial
from scipy.ndimage import median_filter
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from scipy.signal import windows
def get_robust_freq(y_detrended, x_coords, min_cycles=1):
    N = len(y_detrended)
    x_span = x_coords[-1] - x_coords[0]

    min_allowed_freq = min_cycles / x_span if x_span > 0 else 1.0
    
    if N < 4:
        return min_allowed_freq
    dx = x_span / (N - 1)
    
    win = windows.hann(N)
    if N > 30:
        win = windows.hann(N)
        y_windowed = y_detrended * win
    else:
        y_windowed = y_detrended
    
    padding = max(1024, N * 4)
    fourier = np.fft.rfft(y_windowed, n=padding)
    freqs = np.fft.rfftfreq(padding, d=dx)
    magnitudes = np.abs(fourier)
    valid_indices = np.where(freqs >= min_allowed_freq)[0]
    
    if len(valid_indices) == 0:
        return float(min_allowed_freq)
        
    valid_magnitudes = magnitudes[valid_indices]
    peak_idx = np.argmax(valid_magnitudes)
    
    return float(freqs[valid_indices[peak_idx]])
def calculate_floor_sin(batch_segments_x,batch_segments_y):
    x_range=(batch_segments_x[-1]-batch_segments_x[0])
    if np.sign(x_range)==0:
        x_range=1.0
    c1_flat=(batch_segments_y[-1]-batch_segments_y[0])/x_range
    trend_guess_flat=(c1_flat * batch_segments_x)+batch_segments_y[0]
    noise_floor = batch_segments_y - trend_guess_flat
    noise_floor=np.std(noise_floor)
    noise_floor=np.where(noise_floor<1e-9, 1e-9, noise_floor)
    span_batches=max((max(batch_segments_y)-min(batch_segments_y))/2.0,1e-12)
    return noise_floor,span_batches
def initial_guesses_sin(x_dataset,y_dataset,batch_segments_x,batch_segments_y,dataset_std, segment_x, segment_y, segment_index, mode_index,last_mode,parameters_configuration=7):
    noise_floor,noise_floor_span=calculate_floor_sin(batch_segments_x,batch_segments_y)
    segment_span_y=max(segment_y)-min(segment_y)
    x_span=segment_x[-1]-segment_x[0]
    safe_span = max(segment_span_y, 1e-12)

    n_buffer = max(1, len(segment_y) // 10) 

    y_start_stable = np.mean(segment_y[:n_buffer])
    y_end_stable = np.mean(segment_y[-n_buffer:])
    
    c1_start = (y_end_stable - y_start_stable) / x_span

    c0_start = y_start_stable
    
    trend_guess = c1_start * segment_x + c0_start
    y_detrended = segment_y - trend_guess
    y_detrended_std=np.std(y_detrended)

    y_detrended_std_floor=noise_floor

    if np.std(segment_y)<dataset_std*0.1:
        a0_start = noise_floor_span
    else:
        a0_start = y_detrended_std * 1.414
    a1_start = 1e-12 
    
    a0_start = max(a0_start, 0.0)
    x_span_abs=abs(x_span)
    
    b0_start = get_robust_freq(y_detrended, segment_x)
    b0_start = 2.0 * np.pi * b0_start
    estimated_b0 = b0_start
    dynamic_b0_min = estimated_b0 * 0.5
    
    dynamic_b0_max = estimated_b0 * 1.5
    dynamic_b1_max = abs((b0_start * 0.2) / x_span_abs)
    dynamic_b1_min = -dynamic_b1_max
    b1_start = 1e-9

    initial_amplitude = a1_start * segment_x[0] + a0_start
    
    if abs(initial_amplitude) < 1e-12:
        d0_start = 1e-1
    
    else:
        y0_normalized = y_detrended[0] / initial_amplitude
    
        y0_clipped = np.clip(y0_normalized, -1.0, 1.0)
    
        d0_start = np.arcsin(y0_clipped)
              
    safe_a0_max = noise_floor_span*2.0
    safe_a0_min = min(a0_start,y_detrended_std) * 0.7

    safe_a1_max = abs(safe_a0_max / x_span)*0.5
    
    safe_a1_min = -safe_a1_max
    
    c0_min=min(segment_y)-safe_span*0.2
    c0_max=max(segment_y)+safe_span*0.2
    slope_bound = max(np.abs(noise_floor_span/x_span)*2.0,1e-12)
    if last_mode:
        dynamic_b0_min=1e-12
        dynamic_b0_max=b0_start*2.0
        a0_start=safe_span
        safe_a0_max=safe_span*10.0
        c0_min=min(segment_y)-safe_span
        c0_max=max(segment_y)+safe_span
    original_p0=[]
    lower_bounds2=[]
    upper_bounds2=[]
    
    if parameters_configuration==7: #a1,a0,b1,b0,c1,c0,D
        original_p0 = np.array([a1_start, a0_start, b1_start, b0_start, c1_start, c0_start, d0_start])
        lower_bounds2 = [safe_a1_min, safe_a0_min, dynamic_b1_min, dynamic_b0_min, -slope_bound, c0_min,-np.pi]
        upper_bounds2 = [safe_a1_max, safe_a0_max,dynamic_b1_max, dynamic_b0_max,slope_bound,c0_max, np.pi]
    elif parameters_configuration==6: #a1,a0,b0,c1,c0,D
        original_p0 = np.array([a1_start, a0_start, b0_start, c1_start, c0_start, d0_start])
        lower_bounds2 = [safe_a1_min,safe_a0_min, dynamic_b0_min, -slope_bound, c0_min,-np.pi]
        upper_bounds2 = [safe_a1_max,safe_a0_max,dynamic_b0_max,slope_bound,c0_max, np.pi]
    elif parameters_configuration==5: #a0,b0,c1,c0,D
        original_p0 = np.array([a0_start, b0_start, c1_start, c0_start, d0_start])
        lower_bounds2 = [safe_a0_min, dynamic_b0_min, -slope_bound, c0_min,-np.pi]
        upper_bounds2 = [safe_a0_max,dynamic_b0_max,slope_bound,c0_max, np.pi]
    elif parameters_configuration==4: #a0,b0,c0,D
        original_p0 = np.array([a0_start, b0_start, c0_start, d0_start])
        lower_bounds2 = [safe_a0_min, dynamic_b0_min, c0_min,-np.pi]
        upper_bounds2 = [safe_a0_max,dynamic_b0_max, c0_max, np.pi]
    return original_p0,lower_bounds2,upper_bounds2

initial_guess_sin7 = partial(initial_guesses_sin, parameters_configuration=7)
initial_guess_sin6 = partial(initial_guesses_sin, parameters_configuration=6)
initial_guess_sin5 = partial(initial_guesses_sin,parameters_configuration=5)
initial_guess_sin4 = partial(initial_guesses_sin,parameters_configuration=4)

def initial_guess_quadratic(x_dataset,y_dataset,batch_segments_x,batch_segments_y,dataset_std,segment_x, segment_y, segment_index, mode_index, last_mode):
    x_span = segment_x[-1]-segment_x[0]
    n_buffer = max(1, len(segment_y) // 10) 

    y_start_stable = np.mean(segment_y[:n_buffer])
    y_end_stable = np.mean(segment_y[-n_buffer:])
    
    y_span=y_end_stable - y_start_stable
    y_span_floor=max(max(batch_segments_y)-min(batch_segments_y),1e-12)
    if x_span == 0:
        x_span = 1.0 
    
    unit_b = y_span / (x_span**2)
    unit_c = y_span / x_span

    unit_b_floor = y_span_floor / (x_span**2)
    unit_c_floor = y_span_floor / x_span
    
    params_initial_guesses = [unit_b, unit_c, y_start_stable]
    
    lower_bounds = [
        -100 * np.abs(unit_b_floor),            
        -100 * np.abs(unit_c_floor),            
        np.min(segment_y) - np.abs(y_span_floor)
    ]
    
    upper_bounds = [
        100 * np.abs(unit_b_floor),            
        100 * np.abs(unit_c_floor),            
        np.max(segment_y) + np.abs(y_span_floor)
    ]

    return params_initial_guesses, lower_bounds, upper_bounds
def initial_guess_cubic(x_dataset,y_dataset,batch_segments_x,batch_segments_y,dataset_std,segment_x, segment_y, segment_index, mode_index, last_mode):
    x_span = segment_x[-1]-segment_x[0]
    n_buffer = max(1, len(segment_y) // 10) 

    y_start_stable = np.mean(segment_y[:n_buffer])
    y_end_stable = np.mean(segment_y[-n_buffer:])
    
    y_span=y_end_stable - y_start_stable
    y_span_floor=max(max(batch_segments_y)-min(batch_segments_y),1e-12)
    if x_span == 0:
        x_span = 1.0 
    
    unit_a = y_span / (x_span**3)
    unit_b = y_span / (x_span**2)
    unit_c = y_span / x_span

    unit_a_floor = y_span_floor / (x_span**3)
    unit_b_floor = y_span_floor / (x_span**2)
    unit_c_floor = y_span_floor / x_span
    
    params_initial_guesses = [0.0, unit_b, unit_c, y_start_stable]
    
    lower_bounds = [-100 * np.abs(unit_a_floor), 
        -100 * np.abs(unit_b_floor),            
        -100 * np.abs(unit_c_floor),            
        np.min(segment_y) - np.abs(y_span_floor)
    ]
    
    upper_bounds = [100 * np.abs(unit_a_floor), 
        100 * np.abs(unit_b_floor),            
        100 * np.abs(unit_c_floor),            
        np.max(segment_y) + np.abs(y_span_floor)
    ]

    return params_initial_guesses, lower_bounds, upper_bounds

def initial_guess_decay(x_dataset,y_dataset,batch_segments_x,batch_segments_y,dataset_std,segment_x, segment_y, segment_index, mode_index, last_mode):
    x_span = segment_x[-1]-segment_x[0]
    n_buffer = max(1, len(segment_y) // 10) 

    y_start_stable = np.mean(segment_y[:n_buffer])
    y_end_stable = np.mean(segment_y[-n_buffer:])
    
    y_span=y_end_stable - y_start_stable
    y_span_floor=max(max(batch_segments_y)-min(batch_segments_y),1e-12)

    unit_b=1.0/abs(x_span)
    unit_a = y_span / x_span
    unit_a_floor = y_span_floor / x_span
    params_initial_guesses = [unit_a, unit_b,y_start_stable]
    lower_bounds = [
        -100 * np.abs(unit_a_floor), abs(unit_b)*-100.0,  
        np.min(segment_y) - np.abs(y_span_floor)
    ]
    
    upper_bounds = [
        100 * np.abs(unit_a_floor), abs(unit_b)*100.0,         
        np.max(segment_y) + np.abs(y_span_floor)
    ]

    return params_initial_guesses,lower_bounds,upper_bounds
def initial_guess_relation(x_dataset,y_dataset,batch_segments_x,batch_segments_y,dataset_std,segment_x, segment_y, segment_index, mode_index, last_mode):
    x_span = segment_x[-1]-segment_x[0]
    y_span_floor=max(max(batch_segments_y)-min(batch_segments_y),1e-12)
    n_buffer = max(1, len(segment_y) // 10) 

    y_start_stable = np.mean(segment_y[:n_buffer])
    y_end_stable = np.mean(segment_y[-n_buffer:])
    
    y_span=y_end_stable - y_start_stable
    unit_a = y_span / x_span
    unit_a_floor = y_span / x_span

    params_initial_guesses = [unit_a, y_start_stable, 1.0]

    lower_bounds = [-100 * np.abs(unit_a_floor), np.min(segment_y) - np.abs(y_span_floor), 1e-12] 
    upper_bounds = [100 * np.abs(unit_a_floor), np.max(segment_y) + np.abs(y_span_floor), 100 * np.abs(unit_a_floor)]

    return params_initial_guesses, lower_bounds, upper_bounds
def initial_guess_linear(x_dataset,y_dataset,batch_segments_x,batch_segments_y,dataset_std,segment_x, segment_y, segment_index, mode_index, last_mode):
    x_span = segment_x[-1]-segment_x[0]
    y_span_floor=max(max(batch_segments_y)-min(batch_segments_y),1e-12)
    n_buffer = max(1, len(segment_y) // 10) 

    y_start_stable = np.mean(segment_y[:n_buffer])
    y_end_stable = np.mean(segment_y[-n_buffer:])
    
    y_span=y_end_stable - y_start_stable
    unit_a = y_span / x_span
    unit_a_floor = y_span_floor / x_span
    params_initial_guesses=[unit_a,y_start_stable]
    lower_bounds=[-10 * np.abs(unit_a_floor),np.min(segment_y) - np.abs(y_span_floor)]
    upper_bounds=[10 * np.abs(unit_a_floor),np.max(segment_y) + np.abs(y_span_floor)]

    return params_initial_guesses,lower_bounds,upper_bounds
def initial_guess_gaussian(x_dataset,y_dataset, batch_segments_x,batch_segments_y,dataset_std, segment_x, segment_y, segment_index, mode_index, last_mode):
    x_span = segment_x[-1] - segment_x[0]
    y_min, y_max = min(segment_y), max(segment_y)
    y_span_floor=max(max(batch_segments_y)-min(batch_segments_y),1e-12)
    c0_guess = y_min
    a_guess = y_max - c0_guess 
    
    max_idx = np.argmax(segment_y)
    x0_guess = segment_x[max_idx]
    
    sigma_guess = x_span * 0.15 
    
    a_lower = 1e-9
    a_upper = y_span_floor * 3.0 
    
    x0_lower = x0_guess - (x_span * 0.2)
    x0_upper = x0_guess + (x_span * 0.2)
    
    sample_spacing = x_span / max(len(segment_x) - 1, 1)
    sigma_min = x_span * 0.1 
    sigma_max = x_span * 0.5
    
    sigma_lower = min(sigma_min, sigma_max * 0.9)
    sigma_upper = sigma_max
    c_lower = y_min - y_span_floor * 0.2
    c_upper = y_max + y_span_floor * 0.2

    return [a_guess, x0_guess, sigma_guess, c0_guess], [a_lower, x0_lower, sigma_lower, c_lower], [a_upper, x0_upper, sigma_upper, c_upper]
def initial_guess_logistic(x_dataset, y_dataset,batch_segments_x,batch_segments_y,dataset_std, segment_x, segment_y, segment_index, mode_index, last_mode):
    delta_x = float(segment_x[-1] - segment_x[0])
    if delta_x <= 0: 
        delta_x = 1.0
    y_span_floor=max(max(batch_segments_y)-min(batch_segments_y),1e-12)
    y_min, y_max = np.min(segment_y), np.max(segment_y)
    n_buffer = max(1, len(segment_y) // 10) 

    y_start_stable = np.mean(segment_y[:n_buffer])
    y_end_stable = np.mean(segment_y[-n_buffer:])
    
    y_span=y_end_stable - y_start_stable

    c_guess = float(segment_y[0])
    
    L_guess = y_span
    
    x0_guess = float(np.mean(segment_x))
    
    k_guess = 4.0 / delta_x 
    
    params_initial_guesses = [L_guess, k_guess, x0_guess, c_guess]
    lower_bounds = [
        -y_span_floor * 100.0,          
        1e-3 / delta_x,
        segment_x[0] - delta_x, 
        y_min - y_span_floor
    ]
    
    upper_bounds = [
        y_span_floor * 100.0,           
        100.0 / delta_x,       
        segment_x[-1] + delta_x,
        y_max + y_span_floor
    ]

    return params_initial_guesses, lower_bounds, upper_bounds
def initial_guess_fourier(x_dataset, y_dataset,batch_segments_x,batch_segments_y, dataset_std,segment_x, segment_y, segment_index, mode_index, last_mode):
    n_buffer = max(1, len(segment_y) // 10) 
    noise_floor,noise_floor_span=calculate_floor_sin(batch_segments_x,batch_segments_y)
    noise_floor_span=max(noise_floor_span,1e-9)
    y_start_stable = np.mean(segment_y[:n_buffer])
    y_end_stable = np.mean(segment_y[-n_buffer:])
    
    y_span=y_end_stable - y_start_stable
    x_span = segment_x[-1] - segment_x[0]
    n_buffer = max(1, len(segment_y) // 10)

    c1 = y_span / x_span
    c1_floor=noise_floor_span / x_span
    slope_bound = max(np.abs(c1_floor)*2.0,1e-12)
    c0 = y_start_stable
    
    y_detrended = segment_y - (c1 * segment_x + c0)
    a1=max(noise_floor_span*0.1,1e-12)
    a2=max(noise_floor_span*0.05,1e-12)
    a3=max(noise_floor_span*0.01,1e-12)
    num_cycles = get_robust_freq(y_detrended,segment_x)

    detected_freq = 2.0 * np.pi * num_cycles
    
    b1 = np.pi
    b2 = np.pi * 0.5
    b3 = 0.0
    
    f=detected_freq
    params_initial_guesses=[a1,a2,a3,b1,b2,b3,f,c1,c0]
    lower_bounds=[a1*0.5,a2*0.5,a3*0.5,0.0,0.0,0.0, 1e-12,-slope_bound, min(segment_y)-np.abs(noise_floor_span)*0.2]
    upper_bounds=[noise_floor_span,noise_floor_span*0.5,noise_floor_span*0.25,2.0*np.pi,2.0*np.pi,2.0*np.pi, f*5.0,slope_bound,max(segment_y)+np.abs(noise_floor_span)*0.2]
    return params_initial_guesses,lower_bounds,upper_bounds

initial_guesses_models=[initial_guess_sin7,initial_guess_sin6,initial_guess_sin5,initial_guess_sin4,initial_guess_quadratic,initial_guess_cubic,initial_guess_linear,initial_guess_relation, initial_guess_decay, initial_guess_logistic, initial_guess_fourier,initial_guess_gaussian]