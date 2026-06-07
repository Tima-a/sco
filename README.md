# Segmented Continuous Optimization
Piecewise curve fitting remains an essential approach for the comprehensive analysis of local patterns in non-stationary time-series data. However, traditional regression algorithms primarily focus on linear or polynomial functions, which may be insufficient for analyzing raw signals with oscillatory or transcendental behavior. In this paper, we propose Segmented Continuous Optimization (SCO), a framework that performs piecewise continuous curve fitting on various non-linear models, including trigonometric, polynomial, and exponential. SCO presents a novel signal representation by optimizing a user-defined function in segments with C1 continuity to properly analyze the data’s local and global trends. The framework is tested for accuracy and efficiency across all included models. Finally, we provide examples using velocity and EEG datasets to demonstrate the algorithm’s practical usage in examining signal patterns, optimized parameters, derivatives, and integrals of the final fit.

<img width="4234" height="2434" alt="graph_bitcoin" src="https://github.com/user-attachments/assets/28e556bc-9332-4979-9fa3-20076bf24c32" />
Example of SCO on the Bitcoin dataset using a 5-parameter sine wave function.

# How to run
### Prerequisites
* **Python:** 3.9+
* **Libraries:**
    * `numpy` 2.0.2
    * `jax` 0.4.30
    * `jaxlib` 0.4.30
    * `scipy` 1.13.1
    * `matplotlib` 3.9.4
    * `sympy` 1.14.0
    * `pandas` 2.3.0
    * `python-dateutil` 2.9.0.post0

For quick-start, run this code in bash:
```bash
git clone https://github.com/Tima-a/sco.git
cd sco
pip install -r requirements.txt
python sco_example.py
```

IDE Setup:
Download/Clone the repository to your local machine.
Install dependencies: Run ```pip install -r requirements.txt``` in your IDE's terminal.

## Examples:
Segmented Continuous Optimization provides many default functions such as linear, polynomial, trigonometric (4,5,6,7 parameter sine variations), Gaussian, logistic, exponential, rational, and Fourier wave models, with relevant initial guesses.
Run ```sco_applications.py``` to explore applications and working principle of the SCO algorithm on car velocity and EEG datasets.

# Documentation
Documentation focuses on main functions, working principles, arguments and important details.
Example to run SCO:
```pythonfrom sco_main import SCO
import utility
import utility_guesses
import numpy as np

np.random.seed(112)
#Set datasets
y=np.load(f"test_datasets/cryptocoin_tests/test4.npy")[:800]
x=np.arange(len(y))
sco = SCO(
    x_dataset=x, y_dataset=y,
    model=utility.model_sin5,
    initial_guesses_function=utility_guesses.initial_guess_sin5,
    parallel=True,
    verbose=1
)

# Execute fitting
params = sco.run()

# Extract analytic insights
sco.print_fitted_functions()
fitted_y_values=sco.calculate_y_fit_modes()
derivatives = sco.calculate_derivatives(order=1, print_derivative_formulas=True)
integrals = sco.calculate_integrals(order=1, print_integral_formulas=True)
```

Default provided models:
<details>
<summary><b>Click to expand</b></summary>
   
```python
def model_sin7(x, a1, a0, b1, b0, c1,c0, D):
def model_sin6(x, a1,a0, b0, c1, c0, D):
def model_sin5(x, a0, b0, c1, c0, D):
def model_sin4(x, a0, b0, c0, D):
def model_quadratic(x, a, b, c):
def model_cubic(x, a, b, c,d):
def model_linear(x, a, b):
def model_relation(x, a, b, c):
def model_decay(x, a, b, c):
def model_logistic(x, L, k, x0, c):
def model_fourier(x, a1,a2,a3,ph1, ph2, ph3,f,c1,c0):
def model_gaussian(x, A, x0, sigma,c):
```

</details>
Default provided initial guess functions: 
<details>
<summary><b>Click to expand</b></summary>
   
```python
initial_guess_sin7
initial_guess_sin6
initial_guess_sin5
initial_guess_sin4
initial_guess_quadratic
initial_guess_cubic
initial_guess_linear
initial_guess_relation
initial_guess_decay
initial_guess_logistic
initial_guess_fourier
initial_guess_gaussian
```

</details>

To initialize the SCO class:

```python
def __init__(self, x_dataset=None, y_dataset=None, model=None,
             initial_guesses_function=None, continuity_args=None,
             settings_args=None, optimization_settings_args=None, parallel=True, verbose=0):
```

<details>
<summary><b>Click to expand full documentation for this function</b></summary>
   
**Arguments:**
* **x_dataset (array-like, optional):** Independent variable (time/index).
* **y_dataset (array-like, optional):** Dependent variable (signal).
* **model (callable):** The symbolic model function to fit.
* **initial_guesses_function (callable):** Logic used to generate starting parameters for the optimizer.
* **continuity_args (dict):** Configuration for segment stitching.
   * **custom_fitting (bool):** Use default configurations to fix parameters or provide custom parameters to fix. Defaults to True.
   * **value_parameter_fix (string):** Custom value parameter to fix when automatic_fixing is False
   * **derivative_parameter_fix (string):** Custom derivative parameter to fix when automatic_fixing is False  
   * **value_continuity (bool):** Ensure value continuity between segments.
   * **derivative_continuity (bool):** Ensure derivative continuity between segments.
* **settings_args (dict):** SCO configuration settings.
   * **multi_scale (bool):** Perform a multi-scale analysis of all modes, if False, user has to specify number of segments for single smoothing.
   * **num_segments_single (int):** Number of segments for one resolution smoothing. 
   * **scaling (bool):** Apply standard scaling, defaults to True.
   * **unscaling_function (Callable):** Unscaling function which has to be defined if custom_fitting is used.
   * **requested_modes (int):** Number of modes to decompose. If None, the number of modes is calculated using a logarithmic function.
   * **warmup (bool):** Use warmup. Defaults to True
   * **show_plot (bool):** Show final plot. Defaults to True on verbose > 0
   * **non_uniform (bool):** Use non-uniform segmentation. If True, user has to provide all changepoints for each mode.
   * **changepoints_non_uniform (array-like, optional):** Changepoint indices for non-uniform segmentation. 
   * **uniform_num_segments (array-like, optional):** Custom uniform segmentation for each mode. 
   * **hardware_factor (float):** Multiplier for bucketing factor. Defaults to 1.0
* **optimization_settings_args (dict):** Parameters for the Levenberg-Marquardt solver.
   * **batch_size (int):** Number of segments processed in one batch during Levenberg-Marquardt optimization. Defaults to 5. 
   * **max_iters (int):** Maximum number of iterations Levenberg-Marquardt algorithm can use to find the best fit. Defaults to 500.
   * **ftol/xtol (float):** Convergence tolerances for the optimizer. Default to 1e-3.
* **parallel (bool):** Uses a parallel mode processing. Defaults to True.
* **verbose (int):** Verbosity level (0 for silent, 1 for debug logs and plots, 2 for initial guess plots).

Attributes:
* **number_of_modes (int)** - number of modes that we used for decomposition
* **all_full_initial_guesses (list)** - all initial guesses generated for all segments and modes
* **all_changepoints (list)** - all changepoints for all segments and modes
* **results (dict)** - contains "SRMSE", "RMSE", "Time Took", and "Data scale" results for all segments and modes
* **fitted_parameters_modes (list)** - final optimized parameters for all segments and modes 
</details>

To calculate analytical derivatives, integrals, and y-values of optimized decomposition:
```python
def calculate_derivatives(self, order=1, method='analytic', print_derivative_formulas=False):
def calculate_integrals(self, order=1, method='analytic',print_integral_formulas=False):
"""
order (int): order of derivative\integral to calculate.
method (string): method to calculate derivative\integral values, analytic uses derivative\integral of function, numerical uses Newton\Cumulative sum method.
print_derivative_formulas (bool): Print equation of derivative\integral for each segment and mode.
"""
def calculate_y_fit_modes(self):
```

To print optimized functions for all segments, resolutions, and show decomposition plot:
```python
def print_fitted_functions(self):
def show_plot(self):
```

To set x and y datasets:
```python
def set_data(self, x=None, y=None):
```

To set a new model, unscaling_function is needed only when custom fitting is used:
```python
def set_model(self, model, init_guess_model, unscaling_function=None):
```

To run the SCO runner:
```python
def run(self):
```

The user can control the decimal precision for all SCO outputs, including metrics, execution time, and model parameters, by adjusting the global utility setting:
```python
import utility
utility.GLOBAL_PRECISION=6 #default is 3
```

## How to use a custom model with initial guesses for SCO
This is a very important part as a wrong initial guesses or unstable continuity parameters can cause serious numerical instabilities of the LM algorithm. 
### Using a custom model
To use a custom model with SCO, function has to be written using SymPy and follow a specific format:
```python
import sympy as sp
def poly_func(x, a, b, c):
   return a*x**2+b*x+c
def sine_func(x,a,b,c,phi):
   return a*sp.sin(b*x+phi)+c
```

x argument must be the first argument and strictly written as 'x', parameter names can be different(e.g. L, x0, m).
Function must return only direct expression; anything else will throw an error during sympy to numpy/jax conversion. Please ensure that function is differentiable and relatively stable, which is crucial for LM optimization.
### Initial guesses for the custom model 
For a custom function, there should also be an initial guess generator for LM to work properly. Initial guesses are very important because their quality significantly influences overall success of LM, as it is a local optimizer, not global.
Example of initial guess for cubic model function:
```python
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
```
Arguments must be in the same format like x_dataset,y_dataset,batch_segments_x,batch_segments_y,dataset_std,segment_x, segment_y, segment_index, mode_index, as LM will call the intial guess generator function with these arguments during optimization.
x_dataset and y_dataset are original datasets provided by the user, segment_x is locally translated x-values of current segment, segment_y is y-values of current segment, dataset_std is deviation of y_dataset, segment_index is index of current segment, mode_index is index of current mode, max_mode is the last mode index.

### Setting fixed parameters for custom models
Furthermore, for continuity user has to specify fixed value and derivative parameters. <br>
We strongly recommend setting continuity parameters to offset parameters ($ax+b$), where a is for the derivative continuity and b is for value continuity. Setting other fixed parameters can result in highly unstable equations and overall LM optimization.
For example:
```python
def linear_sine_func(x,a1,a0,b,c1,c0,phi):
   return (a1*x+a0)*sp.sin(b*x+phi)+(c1*x+c0)
```
Here, $c_1$ will be specified as derivative continuity parameter and $c_0$ as value continuity parameter.
### Setting unscaling function
Finally, only if custom fitting is used with scaling, currently user has to define an unscaling function with which parameters will be unscaled. This procedure requires substituting $y_{scaled}$ as $\frac{y - \mu_y}{\sigma_y}$ and $x_{scaled}$ as $\frac{x - x_{start}}{\sigma_x}$, then simplifying it to find equations for unscaling parameters. More information on derivation can be found in research paper methodology part.
Implementation of unscaling function for cubic model:
```python
def unscaling_model_cubic(a, b, c, d, sigma_x, mu_y, sigma_y):
    A_new = (a * sigma_y) / (sigma_x**3)
    B_raw = (b * sigma_y) / (sigma_x**2)
    C_raw = (c * sigma_y) / sigma_x
    
    B_new = B_raw
    C_new = C_raw
    D_new = (d * sigma_y) + mu_y

    return [A_new, B_new, C_new, D_new]
```

### Custom fitting example
Example to set custom model, initial guess, and unscaling function for SCO framework:
```python
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
```
# Implementation Details
The Segmented Continuous Optimization (SCO) algorithm balances model complexity and performance. Simple functions can struggle on non-stationary datasets, and complex functions can fit all the noise on most datasets.
### Custom fitting
We highly recommend setting offset parameters($ax+b$) as fixed parameters when using custom fitting. Furthermore, robust initial guess and bounds are very important as tight bounds or bad initial guess can lead to numerical instability and low accuracy. Be careful with setting custom models. For example, running power function($ax^b+c$) will fail if x-dataset contains negative values, as negative number can't be raised to fractional power. Ensure modern version of NumPy is used as 1.x versions can be unstable for SCO framework.
### JAX Compilation
JAX Just-In-Time(JIT) compilation is initial compilation time which is needed to run Segmented Continuous Optimization algorithm. After compilation, algorithm can be run 10x faster if shapes/models didn't change, in case if shape of the datasets, model or some internal parameters change JAX will recompile. Compilation can be done initially by enabling warmup argument in SCO class.

# Future work
Future work will be primarily focused on improving speed and flexiblity of Segmented Continuous Optimization.
Primary tasks are:
1. Increase current speed by improving JAX Levenberg-Marquardt optimization, included initial guesses and convergence tolerances.
2. Expand default prests with more models, initial guesses used in various domains of signal processing.
3. Add a functionality to specify more than two continuity parameters for $C^n$ continuity.
4. Replace or optimize existing SymPy implementations to reduce its overhead.

# Contact
Teymur Aghayev<br>
Email: teymur.aghayev@stud.vilniustech.lt <br>
Linkedin: www.linkedin.com/in/teymur-aghayev-44aa34277

