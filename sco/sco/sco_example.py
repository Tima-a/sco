from sco_main import SCO
import utility
import utility_guesses
import numpy as np
import matplotlib.pyplot as plt

def model_cubic(x, a, b, c,d):
    return a*x**3 + b*x**2 + c*x + d
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

def unscaling_model_cubic(a, b, c, d, sigma_x, mu_y, sigma_y):
    A_new = (a * sigma_y) / (sigma_x**3)
    B_raw = (b * sigma_y) / (sigma_x**2)
    C_raw = (c * sigma_y) / sigma_x
    
    B_new = B_raw
    C_new = C_raw
    D_new = (d * sigma_y) + mu_y

    return [A_new, B_new, C_new, D_new]
y=np.load(f"test_datasets/cryptocoin_tests/test4.npy")[:1400]
x=np.arange(len(y))
continuity={"custom_fitting": True, "value_parameter_fix": 'd',"derivative_parameter_fix": 'c', 'derivative_continuity': True}
settings={"scaling": True, "unscaling_function": unscaling_model_cubic}

SCO = SCO(
    x_dataset=x, y_dataset=y,
    model=model_cubic,
    initial_guesses_function=initial_guess_cubic,continuity_args=continuity, settings_args=settings,
    parallel=True,
    verbose=1
)

params = SCO.run()
SCO.print_fitted_functions()

fitted_y_values=SCO.calculate_y_fit_modes()
derivatives = SCO.calculate_derivatives(order=2, method='numerical')
integrals = SCO.calculate_integrals(order=1, method='numerical')