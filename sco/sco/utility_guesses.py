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
def initial_guesses_sin(x_dataset,y_dataset,batch_segments_x,batch_segments_y,dataset_std, segment_x, segment_y, segment_index, mode_index,parameters_configuration=7):
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

def initial_guess_quadratic(x_dataset,y_dataset,batch_segments_x,batch_segments_y,dataset_std,segment_x, segment_y, segment_index, mode_index):
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
def initial_guess_cubic(x_dataset,y_dataset,batch_segments_x,batch_segments_y,dataset_std,segment_x, segment_y, segment_index, mode_index):
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

def initial_guess_decay(x_dataset, y_dataset, batch_segments_x, batch_segments_y, dataset_std, segment_x, segment_y, segment_index, mode_index):
    x_span = segment_x[-1] - segment_x[0]
    if x_span <= 0:
        x_span = 1e-12
        
    n_buffer = max(1, len(segment_y) // 10) 

    y_start_stable = np.mean(segment_y[:n_buffer])
    y_end_stable = np.mean(segment_y[-n_buffer:])
    
    c_guess = y_end_stable
    a_guess = y_start_stable - y_end_stable
    
    b_guess = 1.0 / x_span 
    
    params_initial_guesses = [a_guess, b_guess, c_guess]
    
    y_span_floor = max(max(batch_segments_y) - min(batch_segments_y), 1e-12)
    
    lower_bounds = [
        -100.0 * y_span_floor,  
        np.abs(b_guess) * -100.0,       
        np.min(segment_y) - y_span_floor  
    ]
    
    upper_bounds = [
        100.0 * y_span_floor,   
        np.abs(b_guess) * 100.0,         
        np.max(segment_y) + y_span_floor  
    ]

    return params_initial_guesses, lower_bounds, upper_bounds
def initial_guess_relation(x_dataset, y_dataset, batch_segments_x, batch_segments_y, dataset_std, segment_x, segment_y, segment_index, mode_index):
    x_span = segment_x[-1] - segment_x[0]
    if x_span <= 0:
        x_span = 1e-12
        
    y_span_floor = max(max(batch_segments_y) - min(batch_segments_y), 1e-12)
    n_buffer = max(1, len(segment_y) // 10) 

    y_start_stable = np.mean(segment_y[:n_buffer])
    y_end_stable = np.mean(segment_y[-n_buffer:])
    y_span = y_end_stable - y_start_stable

    b_guess = y_start_stable
    
    c_guess = 1.0 / x_span
    
    a_guess = y_span * c_guess

    params_initial_guesses = [a_guess, b_guess, c_guess]

    lower_bounds = [
        -100.0 * (y_span_floor / x_span),               
        np.min(segment_y) - y_span_floor,             
        1e-5 / x_span                                   
    ] 
    
    upper_bounds = [
        100.0 * (y_span_floor / x_span),                
        np.max(segment_y) + y_span_floor,               
        100.0 / x_span                               
    ]

    return params_initial_guesses, lower_bounds, upper_bounds
def initial_guess_linear(x_dataset,y_dataset,batch_segments_x,batch_segments_y,dataset_std,segment_x, segment_y, segment_index, mode_index):
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
def initial_guess_gaussian(x_dataset, y_dataset, batch_segments_x, batch_segments_y, dataset_std, segment_x, segment_y, segment_index, mode_index):
    x_span = segment_x[-1] - segment_x[0]
    if x_span <= 0:
        x_span = 1e-12
        
    y_min, y_max = np.min(segment_y), np.max(segment_y)
    y_mean = np.mean(segment_y)
    
    y_span_floor = max(max(batch_segments_y) - min(batch_segments_y), 1e-12)
    
    if np.abs(y_max - y_mean) >= np.abs(y_mean - y_min):
        c0_guess = y_min
        a_guess = y_max - c0_guess
        x0_guess = segment_x[np.argmax(segment_y)]
    else:
        c0_guess = y_max
        a_guess = y_min - c0_guess
        x0_guess = segment_x[np.argmin(segment_y)]

    if np.abs(a_guess) < 1e-12:
        a_guess = 1e-12

    sigma_guess = x_span * 0.15 
    
    params_initial_guesses=[a_guess, x0_guess, sigma_guess, c0_guess]

    lower_bounds = [
        -1.0 * y_span_floor,      
        x0_guess - (x_span * 0.5),
        x_span * 0.01,            
        y_min - y_span_floor * 0.5
    ]
    
    upper_bounds = [
        1.0 * y_span_floor,       
        x0_guess + (x_span * 0.5),
        x_span * 1.5,             
        y_max + y_span_floor * 0.5
    ]

    return params_initial_guesses, lower_bounds, upper_bounds
def initial_guess_logistic(x_dataset, y_dataset,batch_segments_x,batch_segments_y,dataset_std, segment_x, segment_y, segment_index, mode_index):
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
def initial_guess_fourier(x_dataset, y_dataset, batch_segments_x, batch_segments_y, dataset_std, segment_x, segment_y, segment_index, mode_index):
    n_buffer = max(1, len(segment_y) // 10) 
    noise_floor, noise_floor_span = calculate_floor_sin(batch_segments_x, batch_segments_y)
    noise_floor_span = max(noise_floor_span, 1e-9)
    
    y_start_stable = np.mean(segment_y[:n_buffer])
    y_end_stable = np.mean(segment_y[-n_buffer:])
    
    y_span = y_end_stable - y_start_stable
    x_span = segment_x[-1] - segment_x[0]

    c1 = y_span / x_span
    c1_floor = noise_floor_span / x_span
    slope_bound = max(np.abs(c1_floor) * 2.0, 1e-12)
    c0 = y_start_stable
    
    y_detrended = segment_y - (c1 * segment_x + c0)
    f, a1, a2, a3, b1, b2, b3 = analyze_fourier_spectrum(y_detrended, segment_x)

    params_initial_guesses = [a1, a2, a3, b1, b2, b3, f, c1, c0]
    
    lower_bounds = [
        0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 
        1e-12, 
        -slope_bound, 
        min(segment_y) - np.abs(noise_floor_span) * 0.2
    ]
    
    upper_bounds = [
        max(a1 * 5.0, noise_floor_span * 500.0),
        max(a2 * 5.0, noise_floor_span * 200.5),
        max(a3 * 5.0, noise_floor_span * 100.0),
        2.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi, 
        f * 50.0, 
        slope_bound, 
        max(segment_y) + np.abs(noise_floor_span) * 0.2
    ]
    
    return params_initial_guesses, lower_bounds, upper_bounds


def analyze_fourier_spectrum(y_detrended, x_coords, min_cycles=1):
    N = len(y_detrended)
    x_span = x_coords[-1] - x_coords[0]
    min_allowed_freq = min_cycles / x_span if x_span > 0 else 1.0
    
    if N < 4:
        return 2.0 * np.pi * min_allowed_freq, 1e-12, 1e-12, 1e-12, np.pi, np.pi*0.5, 0.0

    dx = x_span / (N - 1)
    
    is_windowed = N > 30
    if is_windowed:
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
        return 2.0 * np.pi * min_allowed_freq, 1e-12, 1e-12, 1e-12, np.pi, np.pi*0.5, 0.0
        
    peak_idx_in_valid = np.argmax(magnitudes[valid_indices])
    idx1 = valid_indices[peak_idx_in_valid] 
    
    idx2 = idx1 * 2
    idx3 = idx1 * 3
    
    amplitude_correction = 4.0 / N if is_windowed else 2.0 / N
    
    a1 = float(magnitudes[idx1] * amplitude_correction)
    a2 = float(magnitudes[idx2] * amplitude_correction) if idx2 < len(fourier) else 1e-12
    a3 = float(magnitudes[idx3] * amplitude_correction) if idx3 < len(fourier) else 1e-12
    
    a1, a2, a3 = max(a1, 1e-12), max(a2, 1e-12), max(a3, 1e-12)

    b1 = (np.angle(fourier[idx1]) + np.pi / 2.0) % (2.0 * np.pi)
    b2 = (np.angle(fourier[idx2]) + np.pi / 2.0) % (2.0 * np.pi) if idx2 < len(fourier) else np.pi * 0.5
    b3 = (np.angle(fourier[idx3]) + np.pi / 2.0) % (2.0 * np.pi) if idx3 < len(fourier) else 0.0
    
    f = float(2.0 * np.pi * freqs[idx1])
    
    return f, a1, a2, a3, b1, b2, b3

initial_guesses_models=[initial_guess_sin7,initial_guess_sin6,initial_guess_sin5,initial_guess_sin4,initial_guess_quadratic,initial_guess_cubic,initial_guess_linear,initial_guess_relation, initial_guess_decay, initial_guess_logistic, initial_guess_fourier,initial_guess_gaussian]