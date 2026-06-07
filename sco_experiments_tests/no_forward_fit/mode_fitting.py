import numpy as np
from scipy.optimize import curve_fit
import time
import jax           
import jax.numpy as jnp 
import concurrent.futures
import utility
import os
from optimizer import lm_fit, lm_start
import utility
from utility import fmt
from collections import namedtuple
import hashlib
import sympy as sp

jax.config.update("jax_enable_x64", True)
class FCD:
    def __init__(self, x_dataset=None, y_dataset=None, model=None, initial_guesses_function=None, continuity_args=None, settings_args=None, optimization_settings_args=None, parallel=True, verbose=0):
        '''
        Args:
            x_dataset (array-like, optional): Independent variable (time/index).
            y_dataset (array-like, optional): Dependent variable (signal).
            model (callable): The symbolic model function to fit.
            initial_guesses_function (callable): Logic used to generate starting parameters for the optimizer.
            continuity_args (dict): Configuration for segment stitching.
                custom_fitting (bool): Use default configurations to fix parameters or provide custom parameters to fix. Defaults to True.
                value_parameter_fix (string): Custom value parameter to fix when automatic_fixing is False
                derivative_parameter_fix (string): Custom derivative parameter to fix when automatic_fixing is False  
                value_continuity (bool): Ensure value continuity between segments.
                derivative_continuity (bool): Ensure derivative continuity between segments.
            settings_args (dict): FCD configuration settings.
                multi_scale (bool): Perform a multi-scale analysis of all modes, if False, user has to specify number of segments for single smoothing.
                num_segments_single (int): Number of segments for single smoothing.
                scaling (bool): Apply standard scaling, defaults to True.
                unscaling_function (Callable): Unscaling function which has to be defined if custom_fitting is used.
                requested_modes (int): Number of modes to decompose. If None, number of modes is calculated using logarithmic equation.
                warmup (bool): Use warmup. Defaults to True
                non_uniform (bool): Use non-uniform segmentation. If true, for multi-scale analysis user has to provide all changepoints for each mode, for single analysis one changepoint array.
                changepoints_non_uniform (array-like, optional): Changepoint indices for non-uniform segmentation. 
                hardware_factor (float): Multiplier for bucketing factor. Defaults to 1.0
            optimization_settings_args (dict): Parameters for the Levenberg-Marquardt solver.
                batch_size (int): Number of segments processed in one batch during Levenberg-Marquardt optimization. Defaults to 5. 
                max_iters (int): Max iterations Levenberg-Marquardt algorithm can use to find best fit. Defaults to 500.
                ftol/xtol (float): Convergence tolerances for the optimizer. Default to 1e-3.
            parallel (bool): Uses a parallel mode processing. Defaults to True.
            verbose (int): Verbosity level (0 for silent, 1 for debug logs and plots, 2 for initial guess plots).
            
        Attributes:
            number_of_modes (int) - number of modes that we used for decomposition
            all_full_initial_guesses (list) - all initial guesses generated for all segments and modes
            all_changepoints (list) - all changepoints for all segments and modes
            results (dict) - contains "SRMSE", "RMSE", "Time Took" and "Data scale" results for all segments and modes
            fitted_parameters_modes (list) - final optimized parameters for all segments and modes 
        '''
        defaults_continuity_args={'custom_fitting': False, 'value_parameter_fix': '', 'derivative_parameter_fix': '', 'value_continuity': True, 'derivative_continuity': True}
        self._continuity_args=defaults_continuity_args
        if continuity_args:
            self._continuity_args.update(continuity_args)

        default_optimization_settings_args={'batch_size': 5, 'max_iters': 500, 'ftol': 1e-3, 'xtol': 1e-3, 'initial_lam': 1e-1, 'bucketing': True}
        self._optimization_settings_args=default_optimization_settings_args
        if optimization_settings_args:
            self._optimization_settings_args.update(optimization_settings_args)

        default_settings_args={'multi_scale': True, 'num_segments_single': 1, 'scaling': True, 'unscaling_function': None, 'requested_modes': None, 'warmup': True, 'non_uniform': False, 'changepoints_non_uniform': None, 'hardware_factor': 1.0}
        self._settings_args=default_settings_args
        if settings_args:
            self._settings_args.update(settings_args)

        self._x_dataset_unscaled = x_dataset
        self._y_dataset_unscaled = y_dataset
        
        self._y_dataset=y_dataset
        self._x_dataset=x_dataset
        if self._settings_args['scaling']:
            self._y_dataset=utility.standard_scaling(y_dataset)
            self._x_dataset=utility.standard_scaling(x_dataset)
        self._model = model
        self._initial_guesses_function = initial_guesses_function
        if not self._continuity_args['custom_fitting'] and self._settings_args['scaling']:
            self._settings_args['unscaling_function']=utility.unscale_map[model]
        self._number_of_modes = self._settings_args['requested_modes']
        self._last_hash=None

        self._verbose=verbose
        self._parallel=parallel
        self._compiled_signatures=[]

        self.parameter_names=None
        self.function_string=""
        self.all_full_initial_guesses=None
        self.all_changepoints=None
        self._fitting_config=None
        self._functions_config=None
        self.results = None
        self.fitted_parameters_modes = None
        if x_dataset is not None and y_dataset is not None:
            self._initialize()
    def _generate_initial_guesses(self):
        all_full_initial_guesses=[]
        all_initial_guesses = []
        all_lower_bounds = []
        all_upper_bounds = []
        dataset_std=np.std(self._y_dataset)
        for mode in range(self._number_of_modes):
            initial_p0s=[]
            initial_lower_bounds=[]
            initial_upper_bounds=[]
            full_initial_p0=[]
            changepoint = np.array(self.all_changepoints[mode])
            changepoints_to_fit = len(changepoint) - 1
            last_mode=False
            if not mode == self._number_of_modes-1 or (not self._settings_args['multi_scale'] and self._settings_args['num_segments_single']>1) or self._settings_args['non_uniform']:
                for i in range(changepoints_to_fit):
                    segment_x = self._x_dataset[changepoint[i]:changepoint[i+1]]-self._x_dataset[changepoint[i]] #zero-centered
                    segment_y = self._y_dataset[changepoint[i]:changepoint[i+1]]
                    initial_p0, initial_lower_bound, initial_upper_bound=self._functions_config.initial_guesses_function(self._x_dataset,self._y_dataset,dataset_std, segment_x,segment_y, i, mode, last_mode)
                    p0_np = np.asarray(initial_p0)
                    lb_np = np.asarray(initial_lower_bound)
                    ub_np = np.asarray(initial_upper_bound)
                    
                    out_of_bounds = (p0_np < lb_np) | (p0_np > ub_np)
                    if out_of_bounds.any() and self._verbose>0:
                        print(f"Warning, initial guess is out of lower and upper bounds for mode {mode} segment {i}, initial guess was clipped")
                    initial_p0=np.clip(initial_p0, initial_lower_bound, initial_upper_bound)
                    if not np.isfinite(np.array(initial_p0)).all() or not np.isfinite(np.array(initial_lower_bound)).all() or not np.isfinite(np.array(initial_upper_bound)).all():
                        raise ValueError("Initial guesses, lower, upper bounds are not finite.")
                    full_initial_p0.append(initial_p0)
                    if i > 0:
                        elements_to_delete=[self._functions_config.index_c0_param,self._functions_config.index_c1_param]
                        if not self._functions_config.derivative_continuity:
                            elements_to_delete=self._functions_config.index_c0_param
                        if self._functions_config.value_continuity:
                            initial_p0=np.delete(initial_p0,elements_to_delete)
                            initial_lower_bound=np.delete(initial_lower_bound,elements_to_delete)
                            initial_upper_bound=np.delete(initial_upper_bound,elements_to_delete)
                    initial_p0s.append(initial_p0)
                    initial_lower_bounds.append(initial_lower_bound)
                    initial_upper_bounds.append(initial_upper_bound)
            else:
                last_mode=True
                segment_x = self._x_dataset-self._x_dataset[0]
                segment_y = self._y_dataset
                initial_p0, initial_lower_bound, initial_upper_bound=self._functions_config.initial_guesses_function(self._x_dataset,self._y_dataset,dataset_std,segment_x, segment_y, 0, mode,last_mode)
                initial_p0=np.clip(initial_p0, initial_lower_bound, initial_upper_bound)
                full_initial_p0 = [initial_p0]
                initial_p0s = initial_p0
                initial_lower_bounds = initial_lower_bound
                initial_upper_bounds = initial_upper_bound
    
    
            all_initial_guesses.append(initial_p0s)
            all_full_initial_guesses.append(full_initial_p0)
            all_lower_bounds.append(initial_lower_bounds)
            all_upper_bounds.append(initial_upper_bounds)
        self.all_full_initial_guesses=all_full_initial_guesses
        self.all_initial_guesses=all_initial_guesses
        self.all_lower_bounds=all_lower_bounds
        self.all_upper_bounds=all_upper_bounds
    def _process_mode(self, mode,x_padded, y_padded):        
        """
        Peforms fitting for each mode, maps parameters into unconstrained space and calculates optimized parameters with Levenberg-Marquardt algorithm.
    
        Args:
            x_padded (list[float]): padded list of x-dataset
            y_padded (list[float]): padded list of y-dataset
    
        Returns:
            dict: {mode index, full parameters for all segments}
        """
        if not mode == self._number_of_modes - 1 or (not self._settings_args['multi_scale'] and self._settings_args['num_segments_single']>1) or self._settings_args['non_uniform']:
            changepoints_to_fit = len(self.all_changepoints[mode]) - 1
    
            p0_unconstrained_jax_new = utility.to_unconstrained(self._params_list_batched[mode], self._lower_list_batched[mode], self._upper_list_batched[mode])
            optimized_parameters, _= lm_start(
                p0_unconstrained_jax_new,x_padded, y_padded, self._lower_list_batched[mode], self._upper_list_batched[mode], 
                self._changepoint_list_batched[mode], changepoints_to_fit, self._segment_lengths_batched[mode], 
                self._modes_length_bucketing, self._max_segment_lengths, mode,self._fitting_config,self._functions_config
            )
            optimized_parameters_full = [item for sublist in optimized_parameters for item in sublist]
    
            return { 'mode': mode, 'segment_lists_params': optimized_parameters_full}
        else:
            x_zero_data=self._x_dataset-self._x_dataset[0]
            bounds_trf = (self.all_lower_bounds[-1],self.all_upper_bounds[-1])
            popt, _ = curve_fit(f=self._functions_config.model_py, xdata=x_zero_data, ydata=self._y_dataset, p0=self.all_initial_guesses[-1], bounds=bounds_trf, method='trf',ftol=1e-5,gtol=1e-5,xtol=1e-5, maxfev=5000)
            return {'mode': mode, 'segment_lists_params': [popt]}
    def calculate_derivatives(self, order=1, method='analytic', print_derivative_formulas=False):
        """
        Calculates derivative values for each mode of decomposition.
    
        Args:
            order (int): order of derivative to calculate.
            method (string): method to calculate derivative values, analytic uses derivative of function, numerical uses Newton method.
            print_derivative_formulas (bool): Print equation of derivative for each segment and mode.
    
        Returns:
            derivative_modes (list[float]): derivative values for each mode of decomposition.
        """
        derivatives_modes = []
        if method == 'numerical':
            y_fits = self.calculate_y_fit_modes()
            derivatives_modes = []
            
            for y_full_array in y_fits:
                deriv_vals = y_full_array
                for _ in range(order):
                    deriv_vals = np.gradient(deriv_vals)
                derivatives_modes.append(deriv_vals)
            return derivatives_modes
        func_derivs, deriv_formula,p_symbols = utility.get_analytic_calculus_derivative(self._model, self.parameter_names, order)
        
        for m in range(len(self.fitted_parameters_modes)):
            derivatives_mode = []
            segments_to_fit = len(self.all_changepoints[m]) - 1
            if print_derivative_formulas:
                print(f"Mode {m+1}")
            for s in range(segments_to_fit):
                last_start=0
                if s==segments_to_fit-1:
                    last_start=1
                x_segment = self._x_dataset_unscaled[self.all_changepoints[m][s]:self.all_changepoints[m][s+1]]
                x_centered=x_segment - x_segment[0]
                deriv_vals = func_derivs(x_centered, *self.fitted_parameters_modes[m][s])
                deriv_vals = np.atleast_1d(deriv_vals)
                if deriv_vals.size == 1 and len(x_centered) > 1:
                    deriv_vals = np.full(len(x_centered), deriv_vals[0])
                if print_derivative_formulas:
                    current_params = self.fitted_parameters_modes[m][s]
                    subs_dict = dict(zip(p_symbols, current_params))
                    raw_deriv = deriv_formula.subs(subs_dict)
                    
                    readable_formula = utility.fast_format(raw_deriv, 3).simplify()
                    print(f"Segment {s+1} from x = {fmt(self._x_dataset_unscaled[self.all_changepoints[m][s]])} to {fmt(self._x_dataset_unscaled[self.all_changepoints[m][s+1]-last_start])}:")
                    if not self.all_changepoints[m][s] == 0:
                        readable_formula = readable_formula.subs(sp.Symbol('x'), sp.Symbol(f'(x-{self.all_changepoints[m][s]})'))
                    print(f"f'(x) = {readable_formula}")
                derivatives_mode.extend(deriv_vals)
    
            derivatives_modes.append(np.array(derivatives_mode))
            
        return derivatives_modes
    def calculate_integrals(self, order=1, method='analytic',print_integral_formulas=False):
        """
        Calculates integral values for each mode of decomposition.
    
        Args:
            order (int): order of integral to calculate.
            method (string): method to calculate integral values, analytic uses integral of function, numerical uses cumulative sum method.
            print_integral_formulas (bool): Print equation of integral for each segment and mode.
    
        Returns:
            integral_modes (list[float]): integral values for each mode of decomposition.
        """
        integrals_modes = []
        if method == 'numerical':
            y_fits = self.calculate_y_fit_modes()
            integrals_modes = []
            for y_full_array in y_fits:
                int_vals = y_full_array
                for _ in range(order):
                    int_vals = np.cumsum(int_vals)
                integrals_modes.append(int_vals)
            return integrals_modes
        func_integrals, integral_formula,p_symbols = utility.get_analytic_calculus_integral(self._model, self.parameter_names, order)
        
        for m in range(len(self.fitted_parameters_modes)):
            integrals_mode = []
            cumulative_c = 0.0
            segments_to_fit = len(self.all_changepoints[m]) - 1
            if print_integral_formulas:
                print(f"Mode {m+1}")
            for s in range(segments_to_fit):
                last_start=0
                if s==segments_to_fit-1:
                    last_start=1
                x_segment = self._x_dataset_unscaled[self.all_changepoints[m][s]:self.all_changepoints[m][s+1]]
                x_centered = x_segment - x_segment[0]
                
                raw_segment = func_integrals(x_centered, *self.fitted_parameters_modes[m][s])
                raw_segment = np.atleast_1d(raw_segment)
                if raw_segment.size == 1 and len(raw_segment) > 1:
                    raw_segment = np.full(len(raw_segment), raw_segment[0])
                
                
                stitched_segment = (raw_segment - raw_segment[0]) + cumulative_c
                integrals_mode.extend(stitched_segment)
                cumulative_c = stitched_segment[-1] 
                if print_integral_formulas:
                    current_params = self.fitted_parameters_modes[m][s]
                    subs_dict = dict(zip(p_symbols, current_params))
                    raw_integral = integral_formula.subs(subs_dict)
                    
                    readable_formula = utility.fast_format(raw_integral, 3).simplify()
                    print(f"Segment {s+1} from x = {fmt(self._x_dataset_unscaled[self.all_changepoints[m][s]])} to {fmt(self._x_dataset_unscaled[self.all_changepoints[m][s+1]-last_start])}:")
                    if not self.all_changepoints[m][s] == 0:
                        readable_formula = readable_formula.subs(sp.Symbol('x'), sp.Symbol(f'(x-{self.all_changepoints[m][s]})'))
                    print(f"Integral of f(x) = {readable_formula}")
            integrals_modes.append(np.array(integrals_mode))
            
        return integrals_modes
    def calculate_y_fit_modes(self):
        """
        Calculates y-values for each mode of decomposition.

        Returns:
            y_fit_modes (list[float]): y-values for each mode of decomposition.
        """
        y_fit_modes=[]
        for m in range(len(self.fitted_parameters_modes)):
            y_fit_mode=[]
            segments_to_fit=len(self.all_changepoints[m])-1
            for s in range(segments_to_fit):
                x_segment=self._x_dataset_unscaled[self.all_changepoints[m][s]:self.all_changepoints[m][s+1]]-self._x_dataset_unscaled[self.all_changepoints[m][s]]
                y_fit_segment=self._functions_config.model_py(x_segment, *self.fitted_parameters_modes[m][s])
                y_fit_mode.extend(y_fit_segment)
            y_fit_modes.append(y_fit_mode)
        return y_fit_modes
    def print_fitted_functions(self):
        """
        Prints optimized functions for each segment and mode of decomposition.

        Returns:
            y_fit_modes (list[float]): y-values for each mode of decomposition.
        """
        for m in range(len(self.fitted_parameters_modes)):
            print(f"Mode {m+1}:")
            segments_to_fit=len(self.all_changepoints[m])-1
            for s in range(segments_to_fit):
                last_start=0
                if s==segments_to_fit-1:
                    last_start=1
                segment_function=self.function_string
                for i in range(len(self.parameter_names)):
                    val_str = f"{self.fitted_parameters_modes[m][s][i]:.3g}"
                    segment_function=segment_function.replace(self.parameter_names[i], val_str)
                segment_function = segment_function.replace('sp.', '')
                segment_function = segment_function.replace('+ -', '- ')
                if not self.all_changepoints[m][s] == 0:
                    segment_function = segment_function.replace('x', f'(x-{self.all_changepoints[m][s]})')
                print(f"Segment {s+1} from x = {fmt(self._x_dataset_unscaled[self.all_changepoints[m][s]])} to {fmt(self._x_dataset_unscaled[self.all_changepoints[m][s+1]-last_start])}:")
                print(f"f(x) = {segment_function}")
    def show_plot(self):
        """
        Shows plot of decomposition.
        """
        utility.show_fitting_plot(self._number_of_modes, self.all_changepoints, self._x_dataset_unscaled,self._y_dataset_unscaled,self.fitted_parameters_modes,self.all_full_initial_guesses,self._functions_config)
    def _get_signature(self):
        sig_str = f"{np.shape(self._x_dataset_unscaled)}_{np.shape(self._y_dataset_unscaled)}_{self._model}_{self._initial_guesses_function}_{str(self._continuity_args)}_{str(self._settings_args)}_{str(self._optimization_settings_args)}"
        return hashlib.md5(sig_str.encode()).hexdigest()
    def set_data(self, x=None, y=None):
        """
        Sets x and y data for FCD algorithm.

        Args:
            x (list[float]): x-dataset to perform FCD algorithm.
            y (list[float]): y-dataset to perform FCD algorithm.
        """
        self._x_dataset_unscaled = x
        self._y_dataset_unscaled=y
        if self._settings_args['scaling']:
            self._y_dataset=utility.standard_scaling(y)
            self._x_dataset=utility.standard_scaling(x)
        else:
            self._y_dataset=y
            self._x_dataset=x
    def set_model(self, model, init_guess_model, unscaling_function=None):
        """
        Sets model, initial guess and unscaling function for FCD algorithm.

        Args:
            model (Callable): model to perform FCD algorithm.
            init_guess_model (Callable): initial guess model to perform FCD algorithm.
            unscaling_function (Callable): unscaling function to unscale optimized parameters, required only if using scaling and custom_fitting.
        """
        self._model=model
        self._initial_guesses_function=init_guess_model
        if not self._continuity_args['custom_fitting'] and self._settings_args['scaling']:
            self._settings_args['unscaling_function']=utility.unscale_map[model]
        else:
            self._settings_args['unscaling_function']=unscaling_function
    def update_settings(self, settings_args, optimization_settings_args, continuity_args):
        self._settings_args.update(settings_args)
        self._optimization_settings_args.update(optimization_settings_args)
        self._continuity_args.update(continuity_args)
    def _warmup_jit(self):
        max_padding_params=self._functions_config.MODEL_FULL_PARAMETER_COUNT+self._functions_config.MODEL_REDUCED_PARAMETER_COUNT*self._fitting_config.batch_size
        max_padding_changepoint=self._fitting_config.batch_size+2
        pad_arr=[]
        if self._max_segment_lengths:
            pad_arr=self._max_segment_lengths[-1]
        else:
            pad_arr=len(self._y_dataset)
        y_fit_full=np.pad(self._y_dataset,(0, pad_arr+1), mode='constant', constant_values=0.0)
        x_fit_full=np.pad(self._x_dataset,(0, pad_arr+1), mode='constant', constant_values=0.0)
        x_data_jnp=jnp.array(x_fit_full,dtype=utility.DTYPE)
        y_full_jnp=jnp.array(y_fit_full, dtype=utility.DTYPE)

        for j in range(len(self._modes_length_bucketing)):
            batch_size   = jnp.array(self._fitting_config.batch_size, dtype=jnp.int32)
            lam  = jnp.array(self._fitting_config.initial_lam, dtype=jnp.float64)
            ftol = jnp.array(self._fitting_config.ftol, dtype=jnp.float64)
            xtol = jnp.array(self._fitting_config.xtol, dtype=jnp.float64)

            lm_fit(
                params_init=jnp.zeros(max_padding_params,dtype=utility.DTYPE),
                x_data=x_data_jnp,
                y=y_full_jnp,
                lower=jnp.zeros(max_padding_params,dtype=utility.DTYPE),
                upper=jnp.zeros(max_padding_params,dtype=utility.DTYPE),
                changepoint_jax=jnp.zeros(max_padding_changepoint,dtype=utility.DTYPE),
                batch_index=jnp.array(0,dtype=utility.INTTYPE),
                prev_params=jnp.zeros(self._functions_config.MODEL_FULL_PARAMETER_COUNT), 
                num_segments=jnp.array(100, dtype=utility.INTTYPE), 
                leftover_batch=jnp.array(1, dtype=utility.INTTYPE),batch_std=jnp.array(100, dtype=utility.DTYPE),batch_size=batch_size,
                max_seg_len=self._modes_length_bucketing[j],fitting_config=self._fitting_config,functions_config=self._functions_config,lam=lam,ridge=jnp.array(1e-12,dtype=utility.DTYPE),can_converge=jnp.array(True, dtype=jnp.bool_), ftol=ftol,xtol=xtol
            )
    def _initialize(self):
        time_init=time.perf_counter()
        if self._model==None:
            self._model=utility.model_sin7
            print("Defaulted to linear sine wave as no model was provided")
        self._number_of_modes=self._settings_args['requested_modes']
        framework_output_dict = utility.create_sequential_framework(self._model, self._initial_guesses_function,self._continuity_args['value_continuity'],self._continuity_args['derivative_continuity'], self._continuity_args['custom_fitting'], self._continuity_args['value_parameter_fix'], self._continuity_args['derivative_parameter_fix'])
        self.parameter_names=framework_output_dict['fitting_params']
        self.function_string=utility.get_exact_function_body(self._model)

        framework_output_dict['value_continuity']=self._continuity_args['value_continuity']
        framework_output_dict['initial_guesses_function']=self._initial_guesses_function
        framework_output_dict['bucketing']=self._optimization_settings_args['bucketing']

        functions_config = namedtuple('FunctionsConfig', framework_output_dict.keys())
        functions_config = functions_config(**framework_output_dict)

        fitting_config = namedtuple('FittingConfig', self._optimization_settings_args.keys())
        fitting_config = fitting_config(**self._optimization_settings_args)

        self._fitting_config=fitting_config
        self._functions_config=functions_config
        N_full = len(self._y_dataset)
        if self._settings_args['non_uniform']==False:
            if self._settings_args['multi_scale']:
                num_segments_uniform=utility.modify_uniform_num_segments(N_full)
                if self._number_of_modes==None:
                    self._number_of_modes=len(num_segments_uniform)
                else:
                    self._number_of_modes=self._number_of_modes
                if not self._settings_args['requested_modes']==None:
                    num_segments_uniform = utility.squash_into_modes(num_segments_uniform, self._number_of_modes)
            else:
                self._number_of_modes=1
                num_segments_uniform=[self._settings_args['num_segments_single']]
            all_changepoints=utility.generate_uniform_segmentation(self._number_of_modes,self._y_dataset, num_segments_uniform, self._settings_args['multi_scale'],self._settings_args['num_segments_single'])
            self._num_segments_uniform=num_segments_uniform
        elif self._settings_args['non_uniform']:
            arr_ch=self._settings_args['changepoints_non_uniform']
            if isinstance(arr_ch, list) and len(arr_ch) > 0 and not isinstance(arr_ch[0], list):
                self._settings_args['changepoints_non_uniform']=[self._settings_args['changepoints_non_uniform']]
            self._number_of_modes=len(self._settings_args['changepoints_non_uniform'])
            all_changepoints=self._settings_args['changepoints_non_uniform']

        self.all_changepoints=all_changepoints
        custom_benchmarks = [(51, 0.0399), (221, 0.0760), (1751, 0.43)]
        custom_benchmarks = utility.adjust_by_hardware_bucketing(custom_benchmarks, self._settings_args['hardware_factor'])
        max_segment_lengths,modes_length_bucketing = utility.generate_bucketing(self._number_of_modes, all_changepoints, custom_benchmarks, self._fitting_config, self._settings_args['multi_scale'],self._settings_args['non_uniform'])
        self._max_segment_lengths=max_segment_lengths
        self._modes_length_bucketing=modes_length_bucketing

        hash_text=self._get_signature()
        self._last_hash=hash_text
        if self._settings_args['warmup']:
            self._warmup_jit()

        if self._verbose>0:
            print(f"Initialization finished in {(time.perf_counter()-time_init):.4f}s")
    def _run_initial_functions(self):
        init_time = time.perf_counter()

        hash_text=self._get_signature()
        
        if not self._last_hash==hash_text:
            print(f"JAX compilation is triggered as shapes/variables are different.")
            self._initialize()
            self._compiled_signatures.append(hash_text)
        utility.validate_inputs(self._x_dataset, self._y_dataset, self._number_of_modes, self._model, self._initial_guesses_function, self._optimization_settings_args,self._settings_args, self._continuity_args)

        self._generate_initial_guesses()
        params_list_batched, changepoint_list_batched,lower_list_batched,upper_list_batched,segment_length_list_batched = utility.batch_transformation(self._number_of_modes,self.all_changepoints,self.all_initial_guesses,self.all_lower_bounds,self.all_upper_bounds, self._fitting_config,self._settings_args['multi_scale'],self._settings_args['num_segments_single'], self._settings_args['non_uniform'])

        if self._verbose>0:
            print(f"Running FCD on {utility.get_name(self._model)} with initial guess function {utility.get_name(self._initial_guesses_function)}")
            print(f"Time took for initial guess and batch transformation: {(time.perf_counter()-init_time):.4f}s")
            print(f"Bucketing segment lengths: {self._modes_length_bucketing}")

        self._params_list_batched=params_list_batched
        self._changepoint_list_batched=changepoint_list_batched
        self._lower_list_batched=lower_list_batched
        self._upper_list_batched=upper_list_batched
        self._segment_lengths_batched=segment_length_list_batched
    def run(self):
        """
        Main function which runs all parts of Functional Continuous Decomposition(FCD) algorithm. 
        
        Returns:
            self.fitted_parameters_modes (list[float]): optimized parameters for all segments and modes.
        """        
        if not self._fitting_config:
            print("Warning, _initialize() was not called, initializing automatically.")
            self._initialize()
        start_main_algorithm=time.perf_counter()
        self._run_initial_functions()
        
        pad_arr=[]
        if self._max_segment_lengths:
            pad_arr=self._max_segment_lengths[-1]
        else:
            pad_arr=len(self._y_dataset)
        y_fit_full=np.pad(self._y_dataset,(0, pad_arr+1), mode='constant', constant_values=0.0)
        x_fit_full=np.pad(self._x_dataset,(0, pad_arr+1), mode='constant', constant_values=0.0)
        results_ordered = []
        if self._settings_args['multi_scale']:
            if self._parallel:            
                results_unordered = []
                num_cores = os.cpu_count() or 8
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
                    futures = [executor.submit(self._process_mode, m, x_fit_full, y_fit_full) for m in range(self._number_of_modes)]
                    for future in concurrent.futures.as_completed(futures):
                        results_unordered.append(future.result())
                
                results_ordered = sorted(results_unordered, key=lambda x: x['mode'])
            else:
                for m in range(self._number_of_modes):
                    result = self._process_mode(m,x_fit_full, y_fit_full)
                    results_ordered.append(result)
        else:
            result = self._process_mode(0,x_fit_full, y_fit_full)
            results_ordered.append(result)
        segment_lists_params=utility.get_fit_values(results_ordered)
        time_took=time.perf_counter()-start_main_algorithm
        if self._settings_args['scaling']:
            unscaled_params_fit=utility.unscale_parameters(segment_lists_params, self._x_dataset_unscaled, self._y_dataset_unscaled, self._model, self.all_changepoints, self._continuity_args['custom_fitting'], self._settings_args['unscaling_function'])
            self.fitted_parameters_modes=unscaled_params_fit
            all_full_guesses_unscaled=utility.unscale_parameters(self.all_full_initial_guesses, self._x_dataset_unscaled, self._y_dataset_unscaled, self._model, self.all_changepoints, self._continuity_args['custom_fitting'], self._settings_args['unscaling_function'])
            self.all_full_initial_guesses=all_full_guesses_unscaled
        else:
            self.fitted_parameters_modes=segment_lists_params
        srmse_all,rmse_all,data_scale=utility.get_metrics(self.fitted_parameters_modes,self._x_dataset_unscaled, self._y_dataset_unscaled,self.all_changepoints,self._functions_config,self._verbose)
        self.results={"SRMSE": srmse_all, "RMSE": rmse_all, "Data scale": data_scale, "Time took": time_took}
        
        if self._verbose>0:
            if self._settings_args['non_uniform']==False:
                print(f"Number of segments in each mode: {self._num_segments_uniform}")
            print(f"Decomposition took {time_took:.4f}s")
            utility.show_fitting_plot(self._number_of_modes, self.all_changepoints, self._x_dataset_unscaled,self._y_dataset_unscaled,self.fitted_parameters_modes,self.all_full_initial_guesses,self._functions_config,self._verbose)
        return self.fitted_parameters_modes