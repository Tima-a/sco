import numpy as np
from functools import partial
def initial_guesses_sin(x_dataset,y_dataset,dataset_std,segment_x, segment_y, segment_index, mode_index,last_mode, parameters_configuration=7):
    segment_trend=segment_y[-1]-segment_y[0]
    noise_floor = dataset_std * 1e-1
    noise_floor=np.where(noise_floor<1e-9, 1e-9, noise_floor)
    sign=np.sign(segment_trend)
    if sign==0:
        sign=1
    segment_trend = max(np.abs(segment_trend), noise_floor)*sign
    segment_span_y=max(segment_y)-min(segment_y)
    x_span=segment_x[-1]-segment_x[0]
    safe_span = max(segment_span_y, noise_floor)

    safe_a0_max = safe_span * 0.8
    y_to_x_ratio = max(segment_trend / x_span, 1e-12)       
    n_buffer = max(1, len(segment_y) // 5) 

    y_start_stable = np.mean(segment_y[:n_buffer])
    y_end_stable = np.mean(segment_y[-n_buffer:])
    
    c1_start = (y_end_stable - y_start_stable) / x_span

    c0_start = y_start_stable
    
    trend_guess = c1_start * segment_x + c0_start
    y_detrended = segment_y - trend_guess
    
    a0_start = np.std(y_detrended) * 1.414 
    a1_start = 1e-12 
    
    a0_start = min(a0_start, segment_span_y * 1.1)
    a0_start = max(a0_start, 0.0)
    x_span_abs=abs(x_span)
    max_freq = (2.0 * np.pi) / x_span_abs
    
    zero_crossings = np.where(np.diff(np.sign(y_detrended)))[0]
    num_cycles = len(zero_crossings) / 2.0
    detected_freq = (2 * np.pi * num_cycles) / x_span_abs
    
    b0_start = np.clip(detected_freq, (2 * np.pi / x_span_abs), max_freq * 0.9)
    estimated_b0 = b0_start
    dynamic_b0_min = estimated_b0 * 0.3
    
    dynamic_b0_max = min(estimated_b0 * 1.7, max_freq)
    freq_headroom = max_freq - b0_start
    dynamic_b1_max = max(abs(freq_headroom / (5.0 * x_span)),1e-12)
    dynamic_b1_min = -dynamic_b1_max

    b1_start = (dynamic_b1_max+dynamic_b1_min)/2.0
    
    
    initial_amplitude = a1_start * segment_x[0] + a0_start
    
    if abs(initial_amplitude) < 1e-12:
        d0_start = 1e-1
    
    else:
        y0_normalized = y_detrended[0] / initial_amplitude
    
        y0_clipped = np.clip(y0_normalized, -1.0, 1.0)
    
        d0_start = np.arcsin(y0_clipped)
                
    safe_a0_max = safe_span*2.0
    
    safe_a1_max = abs(safe_span / x_span)*2.0
    
    safe_a1_min = -safe_a1_max
    y_to_x_ratio=max(abs(segment_trend/x_span)*2.0, abs(c1_start))
    
    c0_min=min(segment_y)-safe_span*0.2
    c0_max=max(segment_y)+safe_span*0.2
       
    if last_mode:
        b0_start=2 * np.pi / x_span_abs
        dynamic_b0_min=b0_start*0.1
        dynamic_b0_max=b0_start*5.0
        a0_start=safe_span
        safe_a0_max=safe_span*10.0
        c0_min=min(segment_y)-safe_span
        c0_max=max(segment_y)+safe_span

    original_p0=[]
    lower_bounds2=[]
    upper_bounds2=[]
    if parameters_configuration==7: #a1,a0,b1,b0,c1,c0,D
        original_p0 = np.array([a1_start, a0_start, b1_start, b0_start, c1_start, c0_start, d0_start])
        lower_bounds2 = [safe_a1_min, -safe_a0_max, dynamic_b1_min, dynamic_b0_min, -1*y_to_x_ratio, c0_min,-np.pi]
        upper_bounds2 = [safe_a1_max, safe_a0_max,dynamic_b1_max, dynamic_b0_max,y_to_x_ratio,c0_max, np.pi]
    elif parameters_configuration==6: #a1,a0,b0,c1,c0,D
        original_p0 = np.array([a1_start, a0_start, b0_start, c1_start, c0_start, d0_start])
        lower_bounds2 = [safe_a1_min,-safe_a0_max, dynamic_b0_min, -1*y_to_x_ratio, c0_min,-np.pi]
        upper_bounds2 = [safe_a1_max,safe_a0_max,dynamic_b0_max,y_to_x_ratio,c0_max, np.pi]
    elif parameters_configuration==5: #a0,b0,c1,c0,D
        original_p0 = np.array([a0_start, b0_start, c1_start, c0_start, d0_start])
        lower_bounds2 = [-safe_a0_max, dynamic_b0_min, -1*y_to_x_ratio, c0_min,-np.pi]
        upper_bounds2 = [safe_a0_max,dynamic_b0_max,y_to_x_ratio,c0_max, np.pi]
    elif parameters_configuration==4: #a0,b0,c0,D
        original_p0 = np.array([a0_start, b0_start, c0_start, d0_start])
        lower_bounds2 = [-safe_a0_max, dynamic_b0_min, c0_min,-np.pi]
        upper_bounds2 = [safe_a0_max,dynamic_b0_max, c0_max, np.pi]
    return original_p0,lower_bounds2,upper_bounds2

initial_guess_sin7 = partial(initial_guesses_sin, parameters_configuration=7)
initial_guess_sin6 = partial(initial_guesses_sin, parameters_configuration=6)
initial_guess_sin5 = partial(initial_guesses_sin, parameters_configuration=5)
initial_guess_sin4 = partial(initial_guesses_sin, parameters_configuration=4)
def initial_guess_quadratic(x_dataset,y_dataset,dataset_std,segment_x, segment_y, segment_index, mode_index, last_mode):
    x_span = segment_x[-1]-segment_x[0]
    y_span = segment_y[-1]-segment_y[0]
    if x_span == 0:
        x_span = 1.0 
    noise_floor = dataset_std * 1e-1
    noise_floor=np.where(noise_floor<1e-9, 1e-9, noise_floor)
    sign=np.sign(y_span)
    if sign==0:
        sign=1
    y_span = max(np.abs(y_span), noise_floor)*sign
    unit_a = y_span / (x_span**2)
    unit_b = y_span / x_span
    params_initial_guesses = [0.0, unit_b, segment_y[0]]
    lower_bounds = [
        -100 * np.abs(unit_a), 
        -100 * np.abs(unit_b),                  
        np.min(segment_y) - np.abs(y_span)
    ]
    
    upper_bounds = [
        100 * np.abs(unit_a), 
        100 * np.abs(unit_b),            
        np.max(segment_y) + np.abs(y_span)
    ]

    return params_initial_guesses, lower_bounds, upper_bounds
def initial_guess_cubic(x_dataset,y_dataset,dataset_std,segment_x, segment_y, segment_index, mode_index, last_mode):
    x_span = segment_x[-1]-segment_x[0]
    y_span = segment_y[-1]-segment_y[0]
    noise_floor = dataset_std * 1e-1
    noise_floor=np.where(noise_floor<1e-9, 1e-9, noise_floor)
    sign=np.sign(y_span)
    if sign==0:
        sign=1
    y_span = max(np.abs(y_span), noise_floor)*sign
    if x_span == 0:
        x_span = 1.0 
    
    unit_a = y_span / (x_span**3)
    unit_b = y_span / (x_span**2)
    unit_c = y_span / x_span
    
    params_initial_guesses = [0.0, unit_b, unit_c, segment_y[0]]
    
    lower_bounds = [-10 * np.abs(unit_a), 
        -10 * np.abs(unit_b),            
        -10 * np.abs(unit_c),            
        np.min(segment_y) - np.abs(y_span)
    ]
    
    upper_bounds = [10 * np.abs(unit_a), 
        10 * np.abs(unit_b),            
        10 * np.abs(unit_c),            
        np.max(segment_y) + np.abs(y_span)
    ]

    return params_initial_guesses, lower_bounds, upper_bounds

def initial_guess_decay(x_dataset,y_dataset,dataset_std,segment_x, segment_y, segment_index, mode_index, last_mode):
    x_span = segment_x[-1]-segment_x[0]
    y_span = segment_y[-1]-segment_y[0]
    unit_b=1/abs(x_span)
    noise_floor = dataset_std * 1e-1
    noise_floor=np.where(noise_floor<1e-9, 1e-9, noise_floor)
    sign=np.sign(y_span)
    if sign==0:
        sign=1
    y_span = max(np.abs(y_span), noise_floor)*sign
    unit_a = y_span / x_span
    params_initial_guesses = [unit_a, unit_b,segment_y[0]]
    lower_bounds = [
        -100 * np.abs(unit_a), abs(unit_b)*-10.0,  
        np.min(segment_y) - np.abs(y_span)
    ]
    
    upper_bounds = [
        100 * np.abs(unit_a), abs(unit_b)*10.0,         
        np.max(segment_y) + np.abs(y_span)
    ]

    return params_initial_guesses,lower_bounds,upper_bounds
def initial_guess_relation(x_dataset,y_dataset,dataset_std,segment_x, segment_y, segment_index, mode_index, last_mode):
    x_span = segment_x[-1]-segment_x[0]
    y_span = segment_y[-1]-segment_y[0]
    noise_floor = dataset_std * 1e-1
    noise_floor=np.where(noise_floor<1e-9, 1e-9, noise_floor)
    sign=np.sign(y_span)
    if sign==0:
        sign=1
    y_span = max(np.abs(y_span), noise_floor)*sign
    unit_a = y_span / x_span

    params_initial_guesses = [unit_a, segment_y[0], 1.0]

    lower_bounds = [-10 * np.abs(unit_a), np.min(segment_y) - np.abs(y_span), 1e-12] 
    upper_bounds = [10 * np.abs(unit_a), np.max(segment_y) + np.abs(y_span), 10 * np.abs(unit_a)]

    return params_initial_guesses, lower_bounds, upper_bounds
def initial_guess_linear(x_dataset,y_dataset,dataset_std,segment_x, segment_y, segment_index, mode_index, last_mode):
    x_span = segment_x[-1]-segment_x[0]
    y_span = segment_y[-1]-segment_y[0]
    noise_floor = dataset_std * 1e-1
    noise_floor=np.where(noise_floor<1e-9, 1e-9, noise_floor)
    sign=np.sign(y_span)
    if sign==0:
        sign=1
    y_span = max(np.abs(y_span), noise_floor)*sign
    unit_a = y_span / x_span
    params_initial_guesses=[unit_a,segment_y[0]]
    lower_bounds=[-10 * np.abs(unit_a),np.min(segment_y) - np.abs(y_span)]
    upper_bounds=[10 * np.abs(unit_a),np.max(segment_y) + np.abs(y_span)]

    return params_initial_guesses,lower_bounds,upper_bounds
def initial_guess_gaussian(x_dataset,y_dataset, dataset_std, segment_x, segment_y, segment_index, mode_index, last_mode):
    x_span = segment_x[-1] - segment_x[0]
    y_min, y_max = np.min(segment_y), np.max(segment_y)
    noise_floor = dataset_std * 1e-1
    noise_floor=np.where(noise_floor<1e-9, 1e-9, noise_floor)
    y_range = np.abs(np.maximum(y_max - y_min, noise_floor))
    c0_guess = y_min
    a_guess = y_max - c0_guess 
    
    max_idx = np.argmax(segment_y)
    x0_guess = segment_x[max_idx]
    
    sigma_guess = x_span * 0.15 
    
    a_lower = 1e-9
    a_upper = y_range * 3.0 
    
    x0_lower = x0_guess - (x_span * 0.2)
    x0_upper = x0_guess + (x_span * 0.2)
    
    sample_spacing = x_span / np.maximum(len(segment_x) - 1, 1)
    sigma_min = x_span * 0.1 
    sigma_max = x_span * 0.5
    
    sigma_lower = np.minimum(sigma_min, sigma_max * 0.9)
    sigma_upper = sigma_max
    c_lower = y_min - y_range * 0.2
    c_upper = y_max - y_range * 0.2

    return [a_guess, x0_guess, sigma_guess, c0_guess], [a_lower, x0_lower, sigma_lower, c_lower], [a_upper, x0_upper, sigma_upper, c_upper]
def initial_guess_logistic(x_dataset, y_dataset,dataset_std, segment_x, segment_y, segment_index, mode_index, last_mode):
    delta_x = float(segment_x[-1] - segment_x[0])
    if delta_x <= 0: 
        delta_x = 1.0
    noise_floor = dataset_std * 1e-1
    noise_floor=np.where(noise_floor<1e-9, 1e-9, noise_floor)
    y_min, y_max = np.min(segment_y), np.max(segment_y)
    y_range = y_max - y_min
    y_range=max(np.abs(y_range),noise_floor)
    c_guess = float(segment_y[0])
    
    L_guess = float(segment_y[-1] - segment_y[0])
    
    x0_guess = float(np.mean(segment_x))
    
    k_guess = 4.0 / delta_x 
    
    params_initial_guesses = [L_guess, k_guess, x0_guess, c_guess]
    lower_bounds = [
        -y_range * 5,          
        1e-3 / delta_x,
        segment_x[0] - delta_x, 
        y_min - y_range
    ]
    
    upper_bounds = [
        y_range * 5,           
        100.0 / delta_x,       
        segment_x[-1] + delta_x,
        y_max + y_range
    ]

    return params_initial_guesses, lower_bounds, upper_bounds
def initial_guess_fourier(x_dataset, y_dataset,dataset_std, segment_x, segment_y, segment_index, mode_index, last_mode):
    y_span = segment_y[-1]-segment_y[0]
    span_x=segment_x[-1]-segment_x[0]
    noise_floor = dataset_std * 1e-1
    noise_floor=np.where(noise_floor<1e-9, 1e-9, noise_floor)
    sign=np.sign(y_span)
    if sign==0:
        sign=1
    y_span = max(np.abs(y_span), noise_floor)*sign
    
    c1=y_span / span_x
    a1=y_span*0.5
    a2=a1/2.0
    a3=a1/3.0
    b1=a1
    b2=a2
    b3=a3
    c0=min(segment_y)
    f=2*np.pi/abs(span_x)
    params_initial_guesses=[a1,a2,a3,b1,b2,b3,f,c1,c0]
    lower_bounds=[np.abs(a1)*-5.0,np.abs(a2)*-5.0,np.abs(a3)*-5.0,np.abs(b1)*-5.0,np.abs(b2)*-5.0,np.abs(b3)*-5.0, 1e-12,np.abs(c1)*-10.0, c0-np.abs(y_span)*0.2]
    upper_bounds=[np.abs(a1)*5.0,np.abs(a2)*5.0,np.abs(a3)*5.0,np.abs(b1)*5.0,np.abs(b2)*5.0,np.abs(b3)*5.0, f*5.0,np.abs(c1)*10.0,max(segment_y)+np.abs(y_span)*0.2]
    return params_initial_guesses,lower_bounds,upper_bounds

initial_guesses_models=[initial_guess_sin7,initial_guess_sin6,initial_guess_sin5,initial_guess_sin4,initial_guess_quadratic,initial_guess_cubic,initial_guess_linear,initial_guess_relation, initial_guess_decay, initial_guess_logistic, initial_guess_fourier,initial_guess_gaussian]