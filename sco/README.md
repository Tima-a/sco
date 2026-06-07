# Functional Continuous Decomposition
The analysis of non-stationary time-series data requires insight into its local and global patterns with physical interpretability. However, traditional smoothing algorithms, such as B-splines, Savitzky-Golay filtering, and Empirical Mode Decomposition (EMD), lack the ability to perform parametric optimization with guaranteed continuity. In this paper, we propose <b> Functional Continuous Decomposition </b> (FCD), a JAX-accelerated framework that performs parametric, continuous optimization on a wide range of mathematical functions. By using Levenberg-Marquardt optimization to achieve up to $C^1$ continuous fitting, FCD transforms raw time-series data into $M$ modes that capture different temporal patterns from short-term to long-term trends. Applications of FCD include physics, medicine, financial analysis, and machine learning, where it is commonly used for the analysis of signal temporal patterns, optimized parameters, derivatives, and integrals of decomposition. Furthermore, FCD can be applied for physical analysis and feature extraction with an average SRMSE of <b> 0.735 </b> per segment and a speed of 0.47s on full decomposition of 1,000 points. Finally, we demonstrate that a Convolutional Neural Network (CNN) enhanced with FCD features, such as optimized function values, parameters, and derivatives, achieved <b> 16.8\% </b> faster convergence and 2.5\% higher accuracy over a standard CNN.

<img width="1809" height="907" alt="Figure_1" src="https://github.com/user-attachments/assets/4cea3d8c-f1df-4127-90d2-3b1d04ba759a" />
Example of Functional Continuous Decomposition on the Bitcoin dataset using a 6-parameter sine wave function.

# How to run
### Prerequisites
* **Python:** 3.9
* **Libraries:**
    * `numpy` 2.0.2
    * `jax` 0.4.30
    * `jaxlib` 0.4.30
    * `scipy` 1.13.1
    * `matplotlib` 3.9.4
    * `sympy` 1.14.0
    * `tensorflow` 2.20.0
    * `scikit-learn` 1.6.1
    * `pandas` 2.3.0
    * `python-dateutil` 2.9.0.post0

For quick-start, run this code in bash:
```bash
git clone https://github.com/Tima-a/fcd.git
cd fcd
pip install -r requirements.txt
python fitting_test.py
```

IDE Setup:
Download/Clone the repository to your local machine.
Install dependencies: Run ```pip install -r requirements.txt``` in your IDE's terminal.

## Examples:
Functional Continuous Decomposition has many default functions, initial guesses, and datasets to run FCD.
The current framework includes linear, quadratic, and cubic polynomials, sinusoidal models (4,5,6,7 parameter variations), decay, Fourier sine series, Gaussian, and logistic functions with relevant initial guesses for them.
Run ```fcd_example.py``` and ```fcd_applications.py``` to explore applications and working principle of the FCD algorithm.

# Core concepts
Functional Continuous Decomposition starts by normalizing the original datasets and performing uniform segmentation for each mode. Modes start from noisy, capturing local patterns, up to higher modes which show global trends.
To ensure each mode is smooth across segment boundaries $x_k$, we enforce $C^0$ (value) and $C^1$ (derivative) continuity by algebraically fixing two parameters. They are solved analytically based on the previous segment's y-value and derivative at the segment boundary. The fixed continuity parameters are calculated from the following equations for segment $k>1$, and the previous segment's local x-value at the segment boundary, denoted as $x_{k-1,l}$:

$$f(0, \mathbf{p_k}) = f(x_{k-1, l}, \mathbf{p_{k-1}})$$

$$f'(0, \mathbf{p_k}) = f'(x_{k-1, l}, \mathbf{p_{k-1}})$$

The first equation is solved for a fixed value parameter, and the second equation for a fixed derivative parameter. In function with linear offset term ($ax+b$), $b$ can be used as a fixed value parameter and $a$ as a fixed derivative parameter for continuity between segments. Furthermore, segments are fitted in batches(default batch size $s$ is 5), modes are fitted in parallel. Forward fit is used to optimize $s+1$ segments within each batch, but the last segment is discarded from the fit; instead, it is assigned as an initial guess for the next batch's first segment. The last segment is discarded specifically to re-optimize it in the next batch while having favorable starting continuity constraints. Thus, during the optimization of one batch, LM enforces continuity from the past segment and ensures overall fit is favorable to the future segment, which efficiently solves error propagation and frequent instability problems.

# Documentation
Documentation focuses on main functions, working principles, arguments and important details.
Example to run FCD:
```python
from mode_fitting import FCD
import utility
import utility_guesses
import numpy as np
import matplotlib.pyplot as plt

#Set datasets
y=np.load(f"test_datasets/test18.npy")[:1000]
x=np.arange(len(y))

#Initialize FCD runner
fcd = FCD(
    x_dataset=x, y_dataset=y,
    model=utility.model_sin6,
    initial_guesses_function=utility_guesses.initial_guess_sin6,
    parallel=True,
    verbose=1
)

# Execute fitting
params = fcd.run()

# Extract analytic insights
fcd.print_fitted_functions()
fitted_y_values=fcd.calculate_y_fit_modes()
derivatives = fcd.calculate_derivatives(order=1, print_derivative_formulas=True)
integrals = fcd.calculate_integrals(order=1)
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
def model_fourier(x, a1,a2,a3,b1,b2,b3,f,c1,c0):
def model_gaussian(x, A, x0, sigma,c):
```

</details>
Default provided initial guess functions: 
<details>
<summary><b>Click to expand</b></summary>
   
```python
initial_guess_sine7
initial_guess_sine6
initial_guess_sine5
initial_guess_sine4
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

To initialize the FCD class:

```python
def __init__(self, x_dataset=None, y_dataset=None, model=None,
            initial_guesses_function=None, continuity_args=None, settings_args=None,
            optimization_settings_args=None, parallel=True, verbose=0):
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
* **settings_args (dict):** FCD configuration settings.
   * **scaling (bool):** Apply standard scaling, defaults to True.
   * **unscaling_function (Callable):** Unscaling function which has to be defined if custom_fitting is used.
   * **requested_modes (int):** Number of modes to decompose. If None, the number of modes is calculated using a logarithmic function.
   * **warmup (bool):** Use warmup. Defaults to True
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

To print optimized functions for all segments, modes, and show decomposition plot:
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

To run the FCD runner:
```python
def run(self):
```

The user can control the decimal precision for all FCD outputs, including metrics, execution time, and model parameters, by adjusting the global utility setting:
```python
import utility
utility.GLOBAL_PRECISION=6 #default is 3
```

## How to set custom models, initial guesses and fixed parameters for FCD
This is a very important part as a wrong initial guesses or unstable fixed parameters can cause serious numerical instabilities of LM algorithm. 
### Using a custom model
To use a custom model with FCD, function has to be written using SymPy and follow a specific format:
```python
import sympy as sp
def poly_func(x, a, b, c):
   return a*x**2+b*x+c
def sine_func(x,a,b,c,phi):
   return a*sp.sin(b*x+phi)+c
```

x argument must be the first argument and strictly written as 'x', parameter names can be different(e.g. L, x0, m).
Function must return only direct expression; anything else will throw an error during sympy to numpy/jax conversion. Please ensure that function is differentiable, which is crucial for LM optimization.
### Initial guesses for the custom model 
For a custom function, there should also be an initial guess generator for LM to work properly. Initial guesses are very important because their quality significantly influences overall success of LM, as it is a local optimizer, not global.
Example of initial guess for cubic model function:
```python
def initial_guess_cubic(x_dataset,y_dataset,dataset_std,segment_x, segment_y, segment_index, mode_index, max_mode):
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
```
Arguments must be in the same format like x_dataset, y_dataset, dataset_std, segment_x, segment_y, segment_index, mode_index, max_mode, as LM will call the intial guess generator function with these arguments during optimization.
x_dataset and y_dataset are original datasets provided by the user, segment_x is locally translated x-values of current segment, segment_y is y-values of current segment, dataset_std is deviation of y_dataset, segment_index is index of current segment, mode_index is index of current mode, max_mode is the last mode index.

### Setting fixed parameters for custom models
Furthermore, for continuity user has to specify fixed value and derivative parameters. <br>
We strongly recommend setting fixed parameters to offset parameters ($ax+b$). Setting other fixed parameters can result in highly unstable equations and overall LM optimization.
For example:
```python
def linear_sine_func(x,a1,a0,b,c1,c0,phi):
   return (a1*x+a0)*sp.sin(b*x+phi)+(c1*x+c0)
```
Here, c1 will be specified as derivative fixed parameter and c0 as value fixed parameter.
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
Example to set custom model, initial guess, and unscaling function for FCD framework:
```python
from mode_fitting import FCD
import utility
import utility_guesses
import numpy as np
import matplotlib.pyplot as plt

def model_linear(x, a, b):
    return a*x + b
def initial_guess_linear(x_dataset,y_dataset,dataset_std,segment_x, segment_y, segment_index, mode_index, max_mode):
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
def unscaling_model_linear(a, b , sigma_x, mu_y, sigma_y):
    A_new = a * sigma_y / sigma_x
    B_new = b * sigma_y + mu_y

    return [A_new, B_new]
y=np.load(f"test_datasets/test3.npy")[:1400]
x=np.arange(len(y))
continuity={"custom_fitting": True, "value_parameter_fix": 'b',"derivative_parameter_fix": '', 'derivative_continuity': False}
settings={"scaling": True, "unscaling_function": unscaling_model_linear}

fcd = FCD(
    x_dataset=x, y_dataset=y,
    model=model_linear,
    initial_guesses_function=initial_guess_linear,continuity_args=continuity, settings_args=settings,
    parallel=True,
    verbose=1
)

params = fcd.run()
fcd.print_fitted_functions()

fitted_y_values=fcd.calculate_y_fit_modes()
derivatives = fcd.calculate_derivatives(order=2, method='numerical')
integrals = fcd.calculate_integrals(order=1, method='numerical')
```
# Implementation Details
The Functional Continuous Decomposition (FCD) algorithm balances model complexity and performance. Simple functions can struggle on non-stationary datasets, and complex functions can fit all noise on most datasets.
### Custom fitting
We highly recommend setting offset parameters($ax+b$) as fixed parameters when using custom fitting. Furthermore, robust initial guess and bounds are very important as tight bounds or bad initial guess can lead to numerical instability and low accuracy. Be careful with setting custom models. For example, running power function($ax^b+c$) will fail if x-dataset contains negative values, as negative number can't be raised to fractional power. Ensure modern version of NumPy is used as 1.x versions can be unstable for FCD framework.
### JAX Compilation
JAX Just-In-Time(JIT) compilation is initial compilation time which is needed to run Functional Continuous Decomposition algorithm. After compilation, algorithm can be run 10x faster if shapes/models didn't change, in case if shape of datasets, model or some internal parameters change JAX will recompile. Compilation can be done initially by enabling warmup argument in FCD class.

# Future work
Future work will be primarily focused on improving speed and flexiblity of Functional Continuous Decomposition.
Primary tasks are:
1. Increase current speed by improving JAX Levenberg-Marquardt optimization, bucketing and convergence rates.
2. Fix numerical instabilities by analyzing the optimization process, unstable functions and implement techniques to solve them.
3. Expand default functions, initial guesses presets with more complex functions used in various domains of signal processing.
4. Add functionality to specify more than two fixed parameters for $C^n$ continuity.
5. Replace or optimize existing SymPy implementations to reduce its overhead.

# Contact
Teymur Aghayev<br>
Email: teymur.aghayev@stud.vilniustech.lt <br>
Linkedin: www.linkedin.com/in/teymur-aghayev-44aa34277

