import numpy as np
import jax           
import jax.numpy as jnp 
from sklearn.metrics import r2_score
import inspect
import sympy as sp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sympy as sp
import inspect
from functools import partial
import matplotlib.ticker as ticker
import re
import utility_guesses
DTYPE=jnp.float64
INTTYPE=jnp.int32
def model_sin7(x, a1, a0, b1, b0, c1,c0, D):
    return (a1*x + a0) * sp.sin((b1*x + b0) * x + D) + (c1*x + c0)
def model_sin6(x, a1,a0, b0, c1, c0, D):
    return (a1*x + a0) * sp.sin(b0*x + D)+(c1*x + c0)
def model_sin5(x, a0, b0, c1, c0, D):
    return a0*sp.sin(b0 * x+D)+(c1*x + c0)
def model_sin4(x, a0, b0, c0, D):
    return a0 * sp.sin(b0 * x + D) + c0
def model_quadratic(x, a, b, c):
    return a*x**2 + b*x + c
def model_cubic(x, a, b, c,d):
    return a*x**3 + b*x**2 + c*x + d
def model_linear(x, a, b):
    return a*x + b
def model_relation(x, a, b, c):
    return (a*x/(c*x + 1))+b
def model_decay(x, a, b, c):
    return a*sp.exp(b*x) + c
def model_logistic(x, L, k, x0, c):
    return L / (1 + sp.exp(-k*(x - x0))) + c
def model_fourier(x, a1,a2,a3,ph1, ph2, ph3,f,c1,c0):
    return a1*sp.sin(f*x + ph1) + a2*sp.sin(2*f*x + ph2) + a3*sp.sin(3*f*x + ph3) + (c1*x + c0)
def model_gaussian(x, A, x0, sigma,c):
    return A * sp.exp(-((x - x0)**2) / (2 * (sigma**2))) + c
def to_constrained_single(pu, lo, up):
    """(-inf, inf) -> [lower, upper] for each segment."""
    sigmoid_like = 0.5 * (np.tanh(pu) + 1.0)
    return lo + (up - lo) * sigmoid_like


def to_unconstrained_single(pc, lo, up):
    """[lower, upper] -> (-inf, inf) for each segment."""
    epsilon = 1e-12
    bounds_range = up - lo
    safe_range = np.where(np.abs(bounds_range) < 1e-15, 1e-15, bounds_range)
    p_norm = (pc - lo) / safe_range
    p_clip = np.clip(p_norm, epsilon, 1.0 - epsilon)
    return np.arctanh(2.0 * p_clip - 1.0)


def to_constrained(params_unconstrained_list, lower_list, upper_list):
    out = []
    for pu, lo, up in zip(params_unconstrained_list, lower_list, upper_list):
        out.append(to_constrained_single(np.asarray(pu),
                                         np.asarray(lo),
                                         np.asarray(up)))
    return out


def to_unconstrained(params_constrained_list, lower_list, upper_list):
    out = []
    for pc, lo, up in zip(params_constrained_list, lower_list, upper_list):
        out.append(to_unconstrained_single(np.asarray(pc),
                                           np.asarray(lo),
                                           np.asarray(up)))
    return out
@jax.jit
def to_constrained_jax(params_unconstrained, lower, upper):
    sigmoid_like = 0.5 * (jnp.tanh(params_unconstrained) + 1.0)
    return lower + (upper - lower) * sigmoid_like

@jax.jit
def to_unconstrained_jax(params_constrained, lower, upper):
    epsilon = 1e-12
    bounds_range = upper - lower
    safe_range = jnp.where(jnp.abs(bounds_range) < 1e-15, 1e-15, bounds_range)
    p_norm = (params_constrained - lower) / safe_range
    p_clip = jnp.clip(p_norm, epsilon, 1.0 - epsilon)
    return jnp.arctanh(2.0 * p_clip - 1.0)

def standard_scaling(dataset, min_deviation=1e-30):
    mean=np.mean(dataset)
    std_dev=np.std(dataset)
    safe_std = max(std_dev, min_deviation)
    dataset_scaled=(dataset-mean)/safe_std
    return dataset_scaled
def unscaling_model_sin7(a1, a0, b1, b0, c1, c0, D, sigma_x, mu_y, sigma_y):
    A1_new = a1 * sigma_y / sigma_x
    C1_new = c1 * sigma_y / sigma_x

    A0_new = a0 * sigma_y 
    C0_new = c0 * sigma_y + mu_y 
    B1_new = b1 / (sigma_x**2)
    B0_new = b0 / sigma_x
    D_new = D 
    
    return [A1_new, A0_new, B1_new, B0_new, C1_new, C0_new, D_new]

def unscaling_model_sin6(a1, a0, b0, c1, c0, D, sigma_x, mu_y, sigma_y):
    A1_new = a1 * sigma_y / sigma_x
    C1_new = c1 * sigma_y / sigma_x
    B0_new = b0 / sigma_x

    A0_new = a0 * sigma_y
    C0_new = c0 * sigma_y + mu_y
    D_new  = D

    return [A1_new, A0_new, B0_new, C1_new, C0_new, D_new]

def unscaling_model_sin5(a0, b0, c1, c0, D, sigma_x, mu_y, sigma_y):
    
    A0_new = a0 * sigma_y
    C1_new = c1 * sigma_y / sigma_x
    B0_new = b0 / sigma_x

    C0_new = c0 * sigma_y + mu_y
    D_new  = D

    return [A0_new, B0_new, C1_new, C0_new, D_new]

def unscaling_model_sin4(a0, b0, c0, D, sigma_x, mu_y, sigma_y):
    
    A0_new = a0 * sigma_y
    C0_new = c0 * sigma_y + mu_y
    B0_new = b0 / sigma_x
    
    D_new = D

    return [A0_new, B0_new, C0_new, D_new]

def unscaling_model_cubic(a, b, c, d, sigma_x, mu_y, sigma_y):
    A_new = (a * sigma_y) / (sigma_x**3)
    B_raw = (b * sigma_y) / (sigma_x**2)
    C_raw = (c * sigma_y) / sigma_x
    
    B_new = B_raw
    C_new = C_raw
    D_new = (d * sigma_y) + mu_y

    return [A_new, B_new, C_new, D_new]

def unscaling_model_quadratic(a, b, c, sigma_x, mu_y, sigma_y):
    A_new = a * sigma_y / (sigma_x**2)
    B_raw = b * sigma_y / sigma_x
    
    B_new = B_raw
    C_new = c * sigma_y + mu_y

    return [A_new, B_new, C_new]

def unscaling_model_linear(a, b, sigma_x, mu_y, sigma_y):
    A_new = a * sigma_y / sigma_x
    
    B_new = b * sigma_y + mu_y

    return [A_new, B_new]
def unscaling_model_decay(a, b, c, sigma_x, mu_y, sigma_y):
    B_new = b / sigma_x
    C_new = c * sigma_y + mu_y
    
    A_new = a * sigma_y 
        
    return [A_new, B_new, C_new]

def unscaling_model_relation(a, b, c, sigma_x, mu_y, sigma_y):
    B_new = (b * sigma_y) + mu_y
    C_new = c / sigma_x
    A_new = (a * sigma_y) / sigma_x
    
    return [A_new, B_new, C_new]

def unscaling_model_logistic(L, k, x0, c, sigma_x, mu_y, sigma_y):
    L_new = L * sigma_y
    K_new = k / sigma_x
    
    C_new = c * sigma_y + mu_y
    X0_new = x0 * sigma_x 

    return [L_new, K_new, X0_new, C_new]

def unscaling_model_gaussian(A, x0, sigma, c, sigma_x, mu_y, sigma_y):
    A_new = A * sigma_y
    Sigma_new = sigma * sigma_x
    
    C_new = c * sigma_y + mu_y
    X0_new = x0 * sigma_x

    return [A_new, X0_new, Sigma_new, C_new]

def unscaling_model_fourier(a1, a2, a3, ph1, ph2, ph3, f, c1, c0, sigma_x, mu_y, sigma_y):    
    F_new = f / sigma_x
    
    C1_new = c1 * sigma_y / sigma_x

    C0_new = c0 * sigma_y + mu_y
    A1_new = a1 * sigma_y
    A2_new = a2 * sigma_y
    A3_new = a3 * sigma_y
    
    ph1_new=ph1
    ph2_new=ph2
    ph3_new=ph3
    return [A1_new, A2_new, A3_new, ph1_new, ph2_new, ph3_new, F_new, C1_new, C0_new]
unscale_map={ model_sin7: unscaling_model_sin7,
             model_sin6: unscaling_model_sin6,
             model_sin5: unscaling_model_sin5,
             model_sin4: unscaling_model_sin4,
             model_cubic: unscaling_model_cubic,
             model_quadratic: unscaling_model_quadratic,
             model_linear: unscaling_model_linear,
             model_relation: unscaling_model_relation,
             model_decay: unscaling_model_decay,
             model_logistic: unscaling_model_logistic,
             model_fourier: unscaling_model_fourier,
             model_gaussian: unscaling_model_gaussian
             }
def unscale_parameters(fitted_parameters_modes, x_dataset, y_dataset, model_sympy,all_changepoints, custom_fitting, unscaling_func):
    unscaled_fitted_params_modes=[]
    mu_x=np.mean(x_dataset)
    sigma_x=np.std(x_dataset)
    mu_y=np.mean(y_dataset)
    sigma_y=np.std(y_dataset)
    num_modes=len(all_changepoints)
    for i in range(len(fitted_parameters_modes)):
        params_mode=[]
        for k in range(len(fitted_parameters_modes[i])):
            if not custom_fitting:
                params_segment = unscale_map[model_sympy](*fitted_parameters_modes[i][k], sigma_x, mu_y, sigma_y)
            else:
                params_segment = unscaling_func(*fitted_parameters_modes[i][k], sigma_x, mu_y, sigma_y)
            params_mode.append(params_segment)
        unscaled_fitted_params_modes.append(params_mode)
    return unscaled_fitted_params_modes
def calculate_r2(y_true, y_pred):
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        return -1e12 

    if len(y_true) < 2: 
        return 0.0
    return r2_score(y_true, y_pred)
def calculate_srmse(y_true, y_pred):
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        return 1e12 

    if len(y_true) < 2: 
        return 0.0
    rmse= calculate_rmse(y_true, y_pred)
    std_dev=max(np.std(y_true),1e-12)
    return rmse/std_dev
def calculate_rmse(y_true, y_pred):
    n = y_true.size
    
    if n == 0: return np.nan

    residuals = y_true - y_pred
    rmse = np.sqrt(np.sum(residuals**2) / n)
    return rmse
def calculate_mae(y_true, y_pred):
    n = y_true.size
    if n == 0: 
        return np.nan

    mae = np.mean(np.abs(y_true - y_pred))
    return mae
GLOBAL_PRECISION=3
def fmt(value, precision=None):
    p = precision if precision is not None else GLOBAL_PRECISION
    return f"{value:.{p}f}"
def validate_inputs(x_data, y_data, requested_modes, model, initial_guesses_function, optimization_settings_args,settings_args,continuity_args):
    if len(x_data) != len(y_data):
        raise ValueError(f"Dataset mismatch: x_data length({len(x_data)}) must equal y_data length({len(y_data)})")
    if len(x_data) < 3 or len(y_data) < 3:
        raise ValueError(f"x or y dataset length can't be less than 3, got ({len(x_data)},{len(y_data)})")
    if not settings_args['multi_scale']:
        if settings_args['num_segments_single']*5.0>len(y_data):
            print(f"Warning: Number of segments is too high ({settings_args['num_segments_single']} segments for {len(y_data)} points)")
        if settings_args['num_segments_single'] < 1:
            raise ValueError(f"Number of segments cannot be lower than 1 (got {settings_args['num_segments_single']})")
    sign_model=inspect.signature(model)
    num_parameters=len(sign_model.parameters)-1 # because x is not counted
    empty_arr=np.arange(10)
    guess, lo, up = initial_guesses_function(empty_arr,empty_arr,empty_arr,empty_arr,0.0,empty_arr,empty_arr,1,1)
    lengths_match = (num_parameters == len(guess) == len(lo) == len(up))
    if not lengths_match:
        raise ValueError(f"Arrays are not the same length for model, initial guess functions or lower, upper bounds")
    chpoints = settings_args.get('changepoints_non_uniform')
    if chpoints is not None:
        if not isinstance(chpoints, list):
            raise ValueError(f"Non-uniform changepoints array must be a list, got {type(chpoints)}")
        if isinstance(chpoints[0], list):
            for m in chpoints:
                if not all(x <= y for x, y in zip(m, m[1:])):
                    raise ValueError("Changepoints must be strictly in non-decreasing order")
                for ch in m:
                    if ch<0 or not isinstance(ch, int):
                        raise ValueError(f"Changepoint indices cannot be negative or float type, got {ch}")
        else:
            if not all(x <= y for x, y in zip(chpoints, chpoints[1:])):
                raise ValueError("Changepoints must be strictly in non-decreasing order")
            for ch in chpoints:
                if ch<0 or not isinstance(ch, int):
                    raise ValueError(f"Changepoint indices cannot be negative or float type, got {ch}")

    min_scale_y=1e-3
    max_scale_y=1e30
    min_scale_x=1e-10
    max_scale_x=1e10
    y_range=np.max(y_data)-np.min(y_data)
    x_range=np.max(x_data)-np.min(x_data)
    if np.array(y_data).size==0:
        raise ValueError("y_dataset is empty. Load data and call _initialize() before run().")
    if np.array(x_data).size==0:
        raise ValueError("x_dataset is empty. Load data and call _initialize() before run().")
    if not settings_args['scaling']:
        if y_range < min_scale_y or y_range > max_scale_y:
            print("Warning, y-dataset scale is outside recommended range for unscaled fitting. Convergence may fail, please enable 'scaling'")
        if x_range < min_scale_x or x_range > max_scale_x:
            print("Warning, x-dataset scale is outside recommended range for unscaled fitting. Convergence may fail, please enable 'scaling'")

    if continuity_args['custom_fitting'] and settings_args['scaling'] and settings_args['unscaling_function'] is None:
        raise ValueError("When using custom model and scaling, unscaling function must be given")
    if optimization_settings_args['batch_size'] < 1:
        raise ValueError(f"Batch length ({optimization_settings_args['batch_size']}) cannot be less than 1")
    if requested_modes is not None:
        if requested_modes<1:
            raise ValueError(f"Requested number of modes cannot be less than 1(got {requested_modes})")
    for name,item in optimization_settings_args.items():
        if item <= 0 and not name == 'bucketing':
            raise ValueError(f"Argument '{name}' must be greater than 0.")
def create_runtime_estimator(benchmarks):
    """
    Creates a runtime estimator using linear interpolation of empirical benchmarks.

    Args:
        benchmarks (list[tuple[int, float]]): List of (maximum segment length, time per batch) tuples obtained from benchmark runs.

    Returns:
        Callable: A function that estimates the runtime (float) for a given segment length.
    """
    if not benchmarks:
        print("Warning: No benchmarks provided. Using default linear estimate (0.0005 * length).")
        return lambda length: 0.0005 * length

    benchmarks.sort(key=lambda x: x[0])
    
    lengths = [b[0] for b in benchmarks]
    times = [b[1] for b in benchmarks]

    interp_func = interp1d(lengths, times, kind='linear', fill_value='extrapolate')
    
    def estimate_runtime(length):
        return max(0.0001, float(interp_func(length)))
    
    return estimate_runtime

def find_optimal_configuration(modes_bucketing_data,compilation_cost=2.3, max_k=10, runtime_benchmarks=None):
    """
    Finds optimal configuration of bucketing maximum segment lengths

    Args:
        modes_bucketing_data (list[tuple[int, int]]): List of (maximum segment length, number of batches) tuples in each mode obtained from benchmark runs.
        compilation_cost (int): JAX compilation time in seconds
        max_k (int): maximum number of bucketing optimized
        runtime_benchmarks (list[tuple[int, float]]): List of (maximum segment length, time per batch) tuples obtained from benchmark runs.

    Returns:
        best_config (dict): Best bucketing configuration dictionary with {number of bucketing values found, bucketing, total time, total compilation time, total runtime}
    """
    runtime_estimator = create_runtime_estimator(runtime_benchmarks or [])
    
    data = sorted(modes_bucketing_data, key=lambda x: x[0])
    n = len(data)
    
    batch_counts = [d[1] for d in data]
    prefix_batches = [0] * (n + 1)
    for i in range(n):
        prefix_batches[i+1] = prefix_batches[i] + batch_counts[i]

    def solve_for_fixed_k(k_target):
        dp = np.full((k_target + 1, n + 1), float('inf'))
        dp[0][0] = 0
        parent = np.zeros((k_target + 1, n + 1), dtype=int)
        
        for b in range(1, k_target + 1):
            for i in range(1, n + 1):
                for j in range(i):
                    bucket_size = data[i-1][0]
                    
                    batches_in_range = prefix_batches[i] - prefix_batches[j]
                    
                    cost_per_batch = runtime_estimator(bucket_size)
                    runtime_cost = batches_in_range * cost_per_batch
                    
                    if dp[b-1][j] + runtime_cost < dp[b][i]:
                        dp[b][i] = dp[b-1][j] + runtime_cost
                        parent[b][i] = j
                        
        buckets = []
        curr_idx = n
        for b in range(k_target, 0, -1):
            prev_idx = parent[b][curr_idx]
            if curr_idx > 0:
                buckets.append(data[curr_idx-1][0])
            curr_idx = prev_idx
            
        return sorted(list(set(buckets))), dp[k_target][n]

    best_config = None
    min_global_time = float('inf')

    for k in range(1, max_k + 1):
        if k > n: break 
        
        buckets, total_runtime = solve_for_fixed_k(k)
        
        current_compile_time = k * compilation_cost
        total_time = current_compile_time + total_runtime
        
        if total_time < min_global_time:
            min_global_time = total_time
            best_config = {
                'optimal_k': k,
                'buckets': buckets,
                'total_time': total_time,
                'compilation_time': current_compile_time,
                'runtime': total_runtime
            }
            
    return best_config
def parse_args(defaults, user_input):
    base = defaults.copy()
    if user_input:
        invalid = [k for k in user_input if k not in base]
        if invalid:
            raise ValueError(f"Invalid keys: {invalid}. Allowed: {list(base.keys())}")
        base.update(user_input)
    return base
def show_fitting_plot(max_mode, all_changepoints, x_data_full_np,y_data_full_np,segment_lists_params,all_full_initial_guesses,functions_config, verbose):
    """
    Shows fitting plot for all modes with original dataset scattered and PELT detected changepoints.
    """
    cols = 1 if max_mode == 1 else 2
    rows = int(np.ceil(max_mode / cols)) 
    figsize=(6 * cols, 4 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, layout='constrained', squeeze=False)
    
    axes_flat = axes.flatten()
    
    for mode in range(max_mode):
        ax = axes_flat[mode]
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
        changepoints_to_fit = len(all_changepoints[mode]) - 1
        changepoint = all_changepoints[mode]
        ax.scatter(x_data_full_np, y_data_full_np, label='Raw Data', s=5, alpha=0.6)
        last_start=0

        for i in range(changepoints_to_fit):
            popt_mode=[]
            if i == changepoints_to_fit-1:
                last_start=1
            popt_list_for_mode = segment_lists_params[mode]
            popt_mode=popt_list_for_mode[i]
            segment_width = x_data_full_np[changepoint[i+1]-last_start] - x_data_full_np[changepoint[i]]
            x_fit_plot = np.linspace(0, segment_width, 100)
            if len(all_changepoints[mode]) < 300: # if first mode has less than 300 segments show lines
                ax.axvline(x=x_data_full_np[changepoint[i]], color='darkslategray', linestyle= '--', linewidth= 1.2, alpha= 0.6)
            
            popt_list_for_mode = segment_lists_params[mode]
            plot_offset = x_data_full_np[changepoint[i]]
            if verbose>1:
                ax.plot(x_fit_plot + plot_offset, 
                        functions_config.model_py(x_fit_plot, *all_full_initial_guesses[mode][i]), 
                        color='green', 
                        linewidth=2)
            ax.plot(x_fit_plot + plot_offset, 
                    functions_config.model_py(x_fit_plot, *popt_mode), 
                    color='red', 
                    linewidth=2, label='Fit')
        if len(all_changepoints[mode]) < 300:
            ax.axvline(x=x_data_full_np[changepoint[-1]-1], color= 'darkslategray', linestyle= '--', linewidth= 1.2, alpha= 0.6)
        ax.set_title(f'Resolution {mode+1} - {changepoints_to_fit} Segments',fontsize=12)
        x_range = abs(x_data_full_np[-1] - x_data_full_np[0])
        x_padding = x_range * 0.02 
        if x_data_full_np[-1] > x_data_full_np[0]:
            ax.set_xlim(x_data_full_np[0] - x_padding, x_data_full_np[-1] + x_padding)
        else:
            ax.set_xlim(x_data_full_np[0] + x_padding, x_data_full_np[-1] - x_padding)
        ax.grid(True, alpha=0.35)

    if max_mode > 1:
        for i in range(max_mode, rows * cols):
            fig.delaxes(axes_flat[i])
    fig.supylabel('Price ($)', fontsize=12)
    fig.supxlabel('Days ($d$)', fontsize=12)
    
    #fig = plt.gcf()
    #fig.set_size_inches(14, 8)  
    #fig.set_dpi(100)          
    #
    #plt.savefig("graph_bitcoin.png", dpi=300, bbox_inches='tight')

    plt.show() 
    plt.close('all')
def squash_into_modes(num_segments_uniform,max_mode):
    """
    Squashes window sizes, minimum segment length, penalty values, EMA window multipliers arrays of length maximum mathematical mode length into length of effective maximum mode by interpolation
    Args:
        num_segments_uniform (list[int]): number of segments in each mode for uniform segmentation.
        max_mode (int): Index of effective maximum mode

    Returns:
        num_segments_uniform (list[int]): squashed number of segments in each mode for uniform segmentation.
    """
    M = len(num_segments_uniform) 
    N = max_mode          
    
    if N >= M:
        return num_segments_uniform

    master_indices = np.arange(M)
    target_indices = np.linspace(0, M - 1, num=N)

    new_num_segments_uniform    = np.interp(target_indices, master_indices, num_segments_uniform)

    num_segments_uniform    = np.round(new_num_segments_uniform).astype(int).tolist()
    

    return num_segments_uniform

def adjust_by_hardware_bucketing(benchmarks, hardware_factor):
    """
    Adjusts reference benchmarks to account for different hardware speeds.

    Args:
        benchmarks (list[tuple]): A list of (max_segment_length, per_batch_speed) representing the reference performance for bucketing.
        hardware_factor (float): Scaling factor. Use > 1.0 for slower hardware and < 1.0 for faster hardware.

    Returns:
        adjusted_benchmarks (list[tuple]): The hardware-adjusted benchmarks used for bucketing 
    """
    adjusted_benchmarks = [(length, speed * hardware_factor) for length, speed in benchmarks]
    return adjusted_benchmarks
def modify_uniform_num_segments(N_full,start_div,end_goal=4, speed=2):
    num_seg=max(N_full//start_div,end_goal)
    num_segments_uniform=[num_seg]
    while num_seg//speed>=end_goal:
        num_seg=max(num_seg//speed,end_goal)
        num_segments_uniform.append(num_seg)
    return num_segments_uniform

def generate_uniform_segmentation(max_mode, y_data_full_np, num_segments_list, multi_scale, num_segments_single):
    all_changepoints = []
    N = len(y_data_full_np)
    
    if multi_scale:
        for i in range(max_mode):
            k=num_segments_list[i]
            cp = np.unique(np.linspace(0, N, k + 1).astype(int))
            
            all_changepoints.append(cp.tolist())
    elif num_segments_single>1:
        cp = np.unique(np.linspace(0, N, num_segments_list[0]+1).astype(int))
        all_changepoints.append(cp.tolist())
    return all_changepoints
def get_metrics(segments_params_modes, x_data_full_np,y_data_full_np,all_changepoints, functions_config, verbose):
    srmse_scores_modes=[]
    rmse_scores_modes=[]
    data_scales_modes=[]
    y_dataset_std=np.std(y_data_full_np)
    flat_count = 0
    for i in range(len(segments_params_modes)):
        params_mode=[]
        changepoint=all_changepoints[i]
        srmse_scores_mode=[]
        rmse_scores_mode=[]
        data_scales_mode=[]
        for k in range(len(segments_params_modes[i])):
            last_start=1
            if k==len(segments_params_modes[i])-1:
                last_start=0
            changepoint_index=k
            x_data_segment=x_data_full_np[changepoint[changepoint_index]:changepoint[changepoint_index+1] + last_start]-x_data_full_np[changepoint[changepoint_index]]
            y_prediction = np.array(functions_config.model_py(x_data_segment, *segments_params_modes[i][k]))
            y_real=np.array(y_data_full_np[changepoint[changepoint_index]:changepoint[changepoint_index+1]+last_start])
            current_srmse=calculate_srmse(y_real, y_prediction)
            current_rmse=calculate_rmse(y_real, y_prediction)
            data_scales_mode.append(max(y_real)-min(y_real))
            
            segment_deviation=max(1e-12,np.std(y_real))
            if segment_deviation<y_dataset_std*1e-2:
                current_srmse=1.0
                flat_count+=1
            
            srmse_scores_mode.append(current_srmse)
            rmse_scores_mode.append(current_rmse)
             
        srmse_scores_modes.append(srmse_scores_mode)
        rmse_scores_modes.append(rmse_scores_mode)
        data_scales_modes.append(data_scales_mode)
    if verbose>1:
        for m in range(len(srmse_scores_modes)):
            print(f"Mode {m+1} results")
            mode_srmse=srmse_scores_modes[m]
            mode_rmse=rmse_scores_modes[m]
            for k in range(len(mode_srmse)):
                last_start=0
                if k==len(mode_srmse)-1:
                    last_start=1
                print(f"Segment {k} from {fmt(x_data_full_np[all_changepoints[m][k]])} to {fmt(x_data_full_np[all_changepoints[m][k+1]-last_start])} finished with SRMSE of {fmt(mode_srmse[k])} and RMSE of {fmt(mode_rmse[k])}")
            if mode_srmse:
                print(f"Max SRMSE of {fmt(np.max(np.array(mode_srmse)))} at segment {mode_srmse.index(np.max(np.array(mode_srmse)))}")
    if verbose>0:
        for i in range(len(srmse_scores_modes)):
            all_srmse_mode=np.array(srmse_scores_modes[i])
            all_rmse_mode=np.array(rmse_scores_modes[i])
            output = (f"Mode {i+1}: Max SRMSE {fmt(np.max(all_srmse_mode))} (with RMSE {fmt(all_rmse_mode[np.argmax(all_srmse_mode)])} at segment {np.argmax(all_srmse_mode)}, "
              f"Max RMSE {fmt(np.max(all_rmse_mode))} (range {fmt(data_scales_modes[i][np.argmax(all_rmse_mode)])}), "
              f"Average SRMSE {fmt(np.mean(np.array(all_srmse_mode)))}, Average RMSE {fmt(np.mean(np.array(all_rmse_mode)))}")
            print(output)
            all_srmse_flat = [item for sublist in srmse_scores_modes for item in sublist]
            all_rmse_flat = [item for sublist in rmse_scores_modes for item in sublist]
        print(f"Average SRMSE across all modes is {fmt(np.mean(np.array(all_srmse_flat)))}")
        print(f"Average RMSE across all modes is {fmt(np.mean(np.array(all_rmse_flat)))}")
    return srmse_scores_modes,rmse_scores_modes,data_scales_modes

def get_name(obj):
    if hasattr(obj, 'func'):
        return obj.func.__name__
    return getattr(obj, '__name__', str(obj))
def get_fit_values(results_ordered):
    """
    Processes all fit values such as full segment parameters, r2 scores, rmse scores, data scales.

    Args:
        results_ordered (list[dict]): List of all mode fits
        
    Returns:
        segment_lists_params (list[float]): list of full parameters for all segments
    """
    current_res=0
    segment_lists_params=[]
    for res in results_ordered:
        segment_lists_params.append(res['segment_lists_params'])
        current_res+=1
    return segment_lists_params

def generate_bucketing(max_mode,all_changepoints, custom_benchmarks, config, multi_scale,non_uniform):
    """
    Generates bucketing list of maximum segment length for optimization

    Args:
        max_mode (int): Index of maximum mode
        all_changepoints (list[int]): list of changepoint indices
        custom_benchmarks (list[tuple[int, float]]): List of (maximum segment length, time per batch) tuples obtained from benchmark runs.
        config (NamedTuple): Stores various static fitting parameters
        verbose (bool): Detailed run bool to show PELT changepoints
    Returns:
        tuple: (max_segment_lengths,modes_length_bucketing)
        max_segment_lengths (list[int]): list of all maximum segment length for each mode
        modes_length_bucketing (list[int]): final bucketing list
    """
    num_batches=[]
    max_segment_lengths=[]

    if multi_scale:
        modes=max_mode
        for m in range(modes):
            changepoints_to_fit=len(all_changepoints[m])-1
            segment_lengths_static_tuple = tuple([all_changepoints[m][s+1] - all_changepoints[m][s] + 1 for s in range(changepoints_to_fit)])
            max_segment_lengths.append(max(segment_lengths_static_tuple))
            leftover_batch=1
            if changepoints_to_fit%config.batch_size==0 or changepoints_to_fit%config.batch_size==1: 
                leftover_batch=0
            num_batches.append(changepoints_to_fit//config.batch_size+leftover_batch)
    else:
        changepoints_to_fit=len(all_changepoints[0])-1
        segment_lengths_static_tuple = tuple([all_changepoints[0][s+1] - all_changepoints[0][s] + 1 for s in range(changepoints_to_fit)])
        max_segment_lengths = [max(segment_lengths_static_tuple)]
        return max_segment_lengths,max_segment_lengths
    if max_segment_lengths==[]:
        return [],[]
    changepoint_bucketing = find_optimal_configuration(
    list(zip(max_segment_lengths, num_batches)), 
    compilation_cost=2.3, 
    max_k=20,
    runtime_benchmarks=custom_benchmarks 
    )
    modes_length_bucketing=changepoint_bucketing['buckets']

    return max_segment_lengths,modes_length_bucketing

def batch_transformation(max_mode, all_changepoints,all_initial_guesses,all_lower_bounds,all_upper_bounds, config,multi_scale, num_segments_single,non_uniform):
    """
    Transforms initial guesses, changepoints, lower, upper bounds, segment lengths arrays into batches.

    Args:
        max_mode (int): Index of maximum mode(effective)
        all_changepoints (list[int]): list of changepoint indices
        all_initial_guesses (list[float]): Initial guesses used in mode fitting for each mode's segments, C1, C0 are removed
        all_lower_bounds (list[float]): Lower bounds for each mode's segments
        all_upper_bounds (list[float]): Upper bounds for each mode's segments
        config (NamedTuple): Stores various static fitting parameters
    Returns:
        tuple (params_list_all, changepoint_list_all,lower_list_all,upper_list_all,segment_length_list_all)
    """
    changepoint_list_all=[]
    params_list_all=[]
    lower_list_all=[]
    upper_list_all=[]
    for mode in range(max_mode):
        changepoints_to_fit=len(all_changepoints[mode])-1
        changepoint_list=[]
        params_list=[]
        lower_list=[]
        upper_list=[]
        changepoint=all_changepoints[mode]
        for f in range(changepoints_to_fit//config.batch_size):
            changepoint_list.append(changepoint[f*config.batch_size:(f+1)*config.batch_size+1]) 
            params_list.append(all_initial_guesses[mode][f*config.batch_size:(f+1)*config.batch_size])
            lower_list.append(all_lower_bounds[mode][f*config.batch_size:(f+1)*config.batch_size])
            upper_list.append(all_upper_bounds[mode][f*config.batch_size:(f+1)*config.batch_size])

        if not changepoints_to_fit%config.batch_size==0:
            changepoint_list.append(changepoint[(changepoints_to_fit//config.batch_size)*config.batch_size:changepoints_to_fit+1]) 
            params_list.append(all_initial_guesses[mode][(changepoints_to_fit//config.batch_size)*config.batch_size:changepoints_to_fit+1]) 
            lower_list.append(all_lower_bounds[mode][(changepoints_to_fit//config.batch_size)*config.batch_size:changepoints_to_fit+1]) 
            upper_list.append(all_upper_bounds[mode][(changepoints_to_fit//config.batch_size)*config.batch_size:changepoints_to_fit+1]) 
        
        params_list = [np.concatenate(sub_list, axis=0) for sub_list in params_list]
        lower_list = [np.concatenate(sub_list, axis=0) for sub_list in lower_list]
        upper_list = [np.concatenate(sub_list, axis=0) for sub_list in upper_list]
        
        changepoint_list_all.append(changepoint_list)
        params_list_all.append(params_list)
        lower_list_all.append(lower_list)
        upper_list_all.append(upper_list)
    return params_list_all, changepoint_list_all,lower_list_all,upper_list_all
def fast_format(expr, precision=3):
    return expr.xreplace({n: sp.Float(n, precision) for n in expr.atoms(sp.Number)})
def get_analytic_calculus_derivative(model_func, parameter_names, order=0):
    x = sp.Symbol('x')
    p = sp.symbols(parameter_names)
    
    expr = model_func(x, *p)
    
    if order > 0:
        calc_expr = sp.diff(expr, x, order)
    if order==0:
        calc_expr = expr
    p_symbols = sp.symbols(parameter_names)
    return sp.lambdify((x, *p), calc_expr, 'numpy'), calc_expr,p_symbols
def get_analytic_calculus_integral(model_func, parameter_names, order=0):
    x = sp.Symbol('x')
    p = sp.symbols(parameter_names)
    expr = model_func(x, *p)
    
    if order > 0:
        calc_expr = expr
        for _ in range(order):
            calc_expr = sp.integrate(calc_expr, x)
    else:
        calc_expr = expr
    p_symbols = sp.symbols(parameter_names)
    return sp.lambdify((x, *p), calc_expr, 'numpy'),calc_expr,p_symbols
default_models={model_sin7: ['c0','c1'],model_sin6: ['c0','c1'],model_sin5: ['c0','c1'],model_sin4: ['c0',''],
                model_quadratic:['c','b'],model_cubic:['d','c'], model_linear:['b',''],model_relation:['b',''],
                model_decay: ['c',''],model_logistic: ['c',''],model_fourier:['c0','c1'],model_gaussian:['c','']}

def get_exact_function_body(user_fn):
    source = inspect.getsource(user_fn)

    match = re.search(r'return\s+(.*)', source)
    if match:
        body = match.group(1).strip()
        return body
    return "Could not parse function body"

def create_sequential_framework(user_fn, initial_guesses_fn,value_continuity,derivative_continuity, custom_fitting, value_param_name='', derivative_param_name=''):
    sig = inspect.signature(user_fn)
    param_names = list(sig.parameters.keys())
    
    symbol_map = {name: sp.Symbol(name) for name in param_names}

    expr = user_fn(**symbol_map)
    if 'x' not in symbol_map:
        raise ValueError("The user function must have 'x' as an argument.")
    fitting_params = [p for p in param_names if p != 'x']

    if custom_fitting==True:
        if value_continuity and not value_param_name in fitting_params:
            raise ValueError(f"No C0 parameter name({value_param_name}) found in function parameters({fitting_params})")
        if derivative_continuity and not derivative_param_name in fitting_params:
            raise ValueError(f"No C1 parameter name({derivative_param_name}) found in function parameters({fitting_params})")
    x = symbol_map['x']
    y_target = sp.Symbol('y_target')
    dy_target = sp.Symbol('dy_target')
    
    f_0 = expr.subs(x, 0)
    if derivative_continuity:
        expr_diff = sp.diff(expr, x)
        f_diff_0 = expr_diff.subs(x, 0)
    if custom_fitting==True and (not initial_guesses_fn in utility_guesses.initial_guesses_models or not user_fn in default_models.keys()):
        print("Warning: When using non-default models or initial guesses, carefully choose the model, continuity parameters, and initial guesses with meaningful upper and lower bounds.")
    if not custom_fitting:     
        value_param_name=default_models[user_fn][0]
        derivative_param_name=default_models[user_fn][1]
        if default_models[user_fn][1]=='':
            derivative_continuity=False
        

    if derivative_continuity:
        if len(fitting_params) < 3:
            raise ValueError("Model Parameters Error: Model must have at least 3 parameters for C1 continuity")
    elif not derivative_continuity and value_continuity:
        if len(fitting_params) < 2:
            raise ValueError("Model Parameters Error: Model must have at least 2 parameters for C0 continuity")
    elif not derivative_continuity and not value_continuity:
        if len(fitting_params) < 1:
            raise ValueError("Model Parameters Error: Model must have at least 1 parameters for C0 continuity")

    if value_continuity:
        if not sp.Symbol(value_param_name) in f_0.free_symbols:
            raise ValueError(f"At f(0) {value_param_name} doesn't exist. Please change fixed parameters. At f(0) value parameter must not depend on derivative and value parameter must be present. In f'(0) derivative parameter must be be present.")
        eq_value = sp.Eq(f_0, y_target)
        chosen_sol_value = sp.solve(eq_value, sp.Symbol(value_param_name))
        chosen_sol_value=chosen_sol_value[0]

    if derivative_continuity:
        if sp.Symbol(derivative_param_name) in f_0.free_symbols:
            raise ValueError(f"Dependency Error: {value_param_name} is affected by {derivative_param_name}. Please change fixed parameters. At f(0) value parameter must not depend on derivative and value parameter must be present. In f'(0) derivative parameter must be be present.")
        if not sp.Symbol(derivative_param_name) in f_diff_0.free_symbols:
                raise ValueError(f"At f'(0) {derivative_param_name} doesn't exist. Please change fixed parameters. At f(0) value parameter must not depend on derivative and value parameter must be present. In f'(0) derivative parameter must be be present.")
        eq_derivative = sp.Eq(f_diff_0, dy_target)
        chosen_sol_derivative = sp.solve(eq_derivative, sp.Symbol(derivative_param_name))
        chosen_sol_derivative=chosen_sol_derivative[0]    
    ordered_params = [symbol_map[p] for p in param_names if p != 'x']
    
    params_for_value = [p for p in ordered_params if p.name not in [value_param_name, derivative_param_name]]
    if not derivative_continuity and value_continuity:
        params_for_value = [p for p in ordered_params if p.name not in [value_param_name]]
    if derivative_continuity:
        params_for_derivative = [p for p in ordered_params if p.name != derivative_param_name]
    if not derivative_continuity and not value_continuity:
        params_for_value = [p for p in ordered_params]

    jax_fn_y_model = sp.lambdify((x, *ordered_params), expr, 'jax')
    jax_fn_y_value=None
    jax_fn_y_value_py=None
    jax_fn_y_derivative=None
    python_deriv_fn=None
    jax_fn_y_derivative_py=None
    if value_continuity:
        jax_fn_y_value = sp.lambdify((y_target, *ordered_params), chosen_sol_value, 'jax')
        jax_fn_y_value_py = sp.lambdify((y_target, *params_for_value), chosen_sol_value, 'numpy')
    else:
        jax_fn_y_value = lambda *args: 0.0
        jax_fn_y_value_py = lambda *args: 0.0
    if derivative_continuity:
        jax_fn_y_derivative = sp.lambdify((dy_target, *ordered_params), chosen_sol_derivative, 'jax')
        python_deriv_fn = sp.lambdify((x, *ordered_params), expr_diff, 'numpy')
        jax_fn_y_derivative_py = sp.lambdify((dy_target, *params_for_derivative), chosen_sol_derivative, 'numpy')
    else:
        jax_fn_y_derivative = lambda *args: 0.0
        python_deriv_fn = lambda *args: 0.0
        jax_fn_y_derivative_py = lambda *args: 0.0
    
    model_py = sp.lambdify((x,*ordered_params), expr, 'numpy')
    
    params_no_x = [p for p in param_names if p != 'x']
    index_to_insert_derivative = 0
    index_derivative_param=0
    index_to_insert_value = 0
    index_value_param=0
    number_of_full_params=len(params_no_x)
    number_of_reduced_params=len(params_no_x)
    indices_params=np.arange(len(params_no_x))
    params_with_value=params_no_x
    if not derivative_continuity and value_continuity:
        params_no_x = [p for p in param_names if p != 'x']
        params_reduced = [p for p in params_no_x if p not in [value_param_name]]
        index_to_insert_value = params_with_value.index(value_param_name)
        index_value_param=params_no_x.index(value_param_name)
        indices_params=np.delete(indices_params, index_value_param)
        number_of_reduced_params-=1
    elif derivative_continuity:
        params_with_value = [p for p in params_no_x if p != derivative_param_name]
        index_to_insert_value = params_with_value.index(value_param_name)
        index_value_param=params_no_x.index(value_param_name)
        index_to_insert_derivative = params_no_x.index(derivative_param_name)
        index_derivative_param=params_no_x.index(derivative_param_name)
        indices_params=np.delete(indices_params, [index_value_param,index_derivative_param])
        number_of_reduced_params-=2
    elif not derivative_continuity and not value_continuity:
        params_with_value = [p for p in params_no_x]
    
    jax_model=partial(jax.jit(jax_fn_y_model))
    jax_fn_y_value=partial(jax.jit(jax_fn_y_value))
    jax_fn_y_derivative=partial(jax.jit(jax_fn_y_derivative))
    jax_model_derivative = partial(jax.jit(jax.grad(jax_model, argnums=0)))
    return {
        'model_py': model_py,
        'model_derivative_py': python_deriv_fn,
        'model_jax': jax_fn_y_model,
        'model_derivative_jax': jax_model_derivative,
        'jax_c0_equation': jax_fn_y_value,
        'jax_c1_equation': jax_fn_y_derivative,
        'index_value_param_insert': index_to_insert_value,
        'index_derivative_param_insert': index_to_insert_derivative,
        'index_c0_param': index_value_param,
        'index_c1_param': index_derivative_param,
        'c0_equation_py': jax_fn_y_value_py,
        'c1_equation_py':jax_fn_y_derivative_py,
        'MODEL_FULL_PARAMETER_COUNT':number_of_full_params,
        'MODEL_REDUCED_PARAMETER_COUNT':number_of_reduced_params,
        'indices_params': tuple(indices_params.tolist()),
        'derivative_continuity': derivative_continuity,
        'fitting_params': tuple(fitting_params),
        }