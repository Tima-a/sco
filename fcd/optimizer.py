import numpy as np
from jax.scipy.linalg import solve
import jax           
import jax.numpy as jnp 
import utility
from functools import partial
from scipy.stats import qmc
DTYPE=utility.DTYPE
INTTYPE=utility.INTTYPE
def unpack_parameters(batch,segments_to_fit, params_reduced, changepoint_list, prev_full_params,x_data_full_np,functions_config):
    """
    Unpacks reduced parameter array into full parameter array by calculating missing C1, C0 parameters from previous batch's parameters except for first batch.    
    C0 and C1 continuity between segments is enforced based on the endpoint and derivative of the previous segment.

    Args:
    batch (int): Current batch index
    segments_to_fit (int): Number of segments to fit in this batch
    params_reduced (list[float]): List containing current batch reduced parameters
    changepoint_list (list[int]): List of changepoint indices
    prev_full_params (list[float]): List of previous batch last segment's full parameters
    x_data_full_np (np.ndarray): List of dataset x-values
    functions_config (NamedTuple): Stores JAX, numpy functions and continuity settings

    Returns:
        full_segments_list(list[float]): Batch full segments parameters
    """
    params_reduced = np.array(params_reduced)
    prev_full_params = np.asarray(prev_full_params)
    changepoint=np.asarray(changepoint_list)
    start_idx=0
    full_segments_list=[]
    if not batch==0:
        full_segments_list=[prev_full_params]
    for j in range(segments_to_fit):
        params_per_segment = functions_config.MODEL_REDUCED_PARAMETER_COUNT
        changepoint_index=j+1
        if batch==0:
            changepoint_index=j
        if batch == 0 and j == 0:
            params_per_segment = functions_config.MODEL_FULL_PARAMETER_COUNT
            p_full_j = params_reduced[start_idx : start_idx + params_per_segment]
        else:
            p_free_j = params_reduced[start_idx : start_idx + params_per_segment]        
            p_full_prev=[]
            params_for_c1=[]
            full_params_segment=[]
            if functions_config.value_continuity:
                p_full_prev = full_segments_list[-1]
                x_end=x_data_full_np[changepoint[changepoint_index]]-x_data_full_np[changepoint[changepoint_index-1]]
                last_point = functions_config.model_py(x_end, *p_full_prev)
                
                c0_fit = functions_config.c0_equation_py(last_point, *p_free_j)
                params_for_c1 = np.insert(p_free_j, functions_config.index_value_param_insert, c0_fit)
            if functions_config.derivative_continuity:
                last_derivative = functions_config.model_derivative_py(x_end, *p_full_prev)
                c1_fit_ = functions_config.c1_equation_py(last_derivative, *params_for_c1)
                full_params_segment=np.insert(params_for_c1, functions_config.index_derivative_param_insert, c1_fit_)
            if functions_config.value_continuity and not functions_config.derivative_continuity:
                full_params_segment=params_for_c1
            if not functions_config.value_continuity and not functions_config.derivative_continuity:
                full_params_segment=p_free_j
            
            p_full_j = np.array(full_params_segment)
        full_segments_list.append(p_full_j)
        start_idx += params_per_segment
    if not batch==0:
        full_segments_list.pop(0)
    return full_segments_list

@partial(jax.jit, static_argnames=["max_seg_len", "fitting_config","functions_config"])
def residuals_next_iterations(optimized_params, x_data_full, y_data_full, changepoint_jax, batch_index, prev_params, num_segments, leftover_batch,max_seg_len, fitting_config,functions_config):
    """
    JAX-accelerated residual function which calculates padded residuals of current batch optimization.

    Args:
        optimized_params (jnp.ndarray): Padded JAX array containing optimized parameters
        x_data_full (jnp.ndarray): Padded JAX array of dataset x-values
        y_data_full (jnp.ndarray): Padded JAX array of dataset y-values
        changepoint_jax (jnp.ndarray): Padded JAX array of changepoint indices
        segment_lengths_jax (jnp.ndarray): Padded JAX array of segment lengths
        batch_index (jnp.int32): Current batch index
        prev_params (jnp.ndarray): JAX array of previous batch last segment's full parameters
        num_segments (jnp.int32): Number of segments to fit in this mode
        leftover_batch (jnp.int32): 1 if a partial batch exists, 0 otherwise.
        max_seg_len (int): Maximum segment length in this mode for residual array padding
        fitting_config (NamedTuple): Stores various static parameters for fitting
        functions_config (NamedTuple): Stores JAX, numpy functions and continuity settings

    Returns:
        collected_residuals(jnp.ndarray): Padded JAX residual array 
    """
    last_batch_index = num_segments // fitting_config.batch_size + leftover_batch - 1
    
    segments_left = num_segments - ((num_segments // fitting_config.batch_size) * fitting_config.batch_size)
    remainder = num_segments % fitting_config.batch_size
    
    leftover_segments_last_batch = jnp.where(
        remainder == 0,
        fitting_config.batch_size,
        jnp.where(
            remainder == 1,
            fitting_config.batch_size+1,
            segments_left
        )
    )
    indices_params_jax = jnp.array(functions_config.indices_params)
    segments_to_fit = jnp.where(batch_index == last_batch_index, leftover_segments_last_batch, fitting_config.batch_size+1)
    def loop_jacobian_next_iterations(state, j_index):
        unpacked_elements_array, segment_start_index = state
        segment_index = j_index
        
        should_run = (batch_index < last_batch_index) | ((batch_index == last_batch_index) & (segment_index < segments_to_fit))
        
        global_first_segment_fit = (batch_index == 0) & (segment_index == 0)
        first_batch_next_segments_fit = (batch_index == 0) & (segment_index > 0)
        
        def get_7_params(operand):
            unpacked_elements_array, segment_start_index, _ = operand
            segment_params = jax.lax.dynamic_slice(optimized_params, (segment_start_index,), (functions_config.MODEL_FULL_PARAMETER_COUNT,))
            
            unpacked_elements_array = unpacked_elements_array.at[segment_index].set(segment_params)
            segment_start_index = segment_start_index + functions_config.MODEL_FULL_PARAMETER_COUNT
            return unpacked_elements_array, segment_start_index

        def get_5_params_and_derive(operand):
            unpacked_elements_array, segment_start_index, next_segment_first_batch = operand
            modified_index=segment_index - next_segment_first_batch
            
            idx_end = (modified_index + 1).astype(jnp.int32)
            idx_start = (modified_index).astype(jnp.int32)
            
            start_pos = changepoint_jax[idx_start].astype(jnp.int32)
            end_pos = changepoint_jax[idx_end].astype(jnp.int32)
            
            x_end_val = x_data_full[end_pos] - x_data_full[start_pos]
            x_end = jnp.asarray(x_end_val, dtype=DTYPE)

            prev_idx = modified_index
            last_point = functions_config.model_jax(x_end, *unpacked_elements_array[prev_idx])
            

            segment_params = jax.lax.dynamic_slice(optimized_params, (segment_start_index,), (functions_config.MODEL_REDUCED_PARAMETER_COUNT,))
            full_params_segment=jnp.zeros(functions_config.MODEL_FULL_PARAMETER_COUNT)
            full_params_segment=full_params_segment.at[indices_params_jax].set(segment_params)

            def calculate_c0(full_params_segment):
                c0_fit = functions_config.jax_c0_equation(last_point, *full_params_segment)
                full_params_segment=full_params_segment.at[functions_config.index_c0_param].set(c0_fit)
                return full_params_segment
            def set_c0(full_params_segment):
                return full_params_segment
            def calculate_c1(full_params_segment):
                last_derivative = functions_config.model_derivative_jax(x_end, *unpacked_elements_array[prev_idx])
                c1_fit_ = functions_config.jax_c1_equation(last_derivative, *full_params_segment)
                full_params_segment=full_params_segment.at[functions_config.index_c1_param].set(c1_fit_)
                return full_params_segment
            def set_c1(full_params_segment):
                return full_params_segment
            full_params_segment = jax.lax.cond(functions_config.value_continuity, calculate_c0, set_c0, operand=full_params_segment)
            full_params_segment = jax.lax.cond(functions_config.derivative_continuity, calculate_c1, set_c1, operand=full_params_segment)
            target_idx = modified_index+1
            unpacked_elements_array = unpacked_elements_array.at[target_idx].set(full_params_segment)
            
            segment_start_index = segment_start_index + functions_config.MODEL_REDUCED_PARAMETER_COUNT
            return unpacked_elements_array, segment_start_index
            
        def change_start_j(_): return jnp.array(1, dtype=INTTYPE)
        def identity_j(_): return jnp.array(0, dtype=INTTYPE)
        
        def run_residuals(carry):
            local_unpacked_elements_array, segment_start_index = carry 
            
            next_segment_first_batch = jax.lax.cond(
                first_batch_next_segments_fit,
                change_start_j,
                identity_j,
                operand=None
            )
            
            new_unpacked_array, new_start_index = jax.lax.cond(
                global_first_segment_fit,
                get_7_params,
                get_5_params_and_derive,
                operand=(local_unpacked_elements_array, segment_start_index, next_segment_first_batch)
            )
            index_offset = jnp.where(batch_index == 0, 0, 1)
            params_index = segment_index + index_offset
            start_index_segment = changepoint_jax[params_index].astype(INTTYPE)
            current_params = new_unpacked_array[params_index]
            
            end_index_segment = changepoint_jax[params_index + 1].astype(INTTYPE)
            segment_length_tracer = (end_index_segment - start_index_segment).astype(INTTYPE)
            x_range_max_sliced = jax.lax.dynamic_slice(x_data_full, (start_index_segment,), (max_seg_len,)) - x_data_full[start_index_segment]

            sequential_indices = jnp.arange(max_seg_len, dtype=INTTYPE)
            mask_float = (sequential_indices < segment_length_tracer).astype(DTYPE)

            x_range_masked = x_range_max_sliced * mask_float
            
            y_prediction = functions_config.model_jax(x_range_masked, *current_params)

            
            segment_y_target_oversliced = jax.lax.dynamic_slice(y_data_full, (start_index_segment,), (max_seg_len,))
            
            y_target_masked = segment_y_target_oversliced * mask_float 

            residual_padded = y_target_masked - y_prediction
            
            residual_padded_masked = residual_padded * mask_float
            
            new_carry = (new_unpacked_array, new_start_index)
            return new_carry, residual_padded_masked

        def return_empty(carry):
            return carry, jnp.zeros(max_seg_len, dtype=DTYPE)

        new_state, residuals = jax.lax.cond(
            should_run,
            run_residuals,
            return_empty,
            operand=(unpacked_elements_array, segment_start_index)
        )

        return new_state, residuals

    unpacked_elements = jnp.zeros((fitting_config.batch_size + 2, functions_config.MODEL_FULL_PARAMETER_COUNT), dtype=DTYPE)
    unpacked_elements = unpacked_elements.at[0].set(prev_params)
    segment_start_index = jnp.array(0, dtype=INTTYPE)

    initial_state = (unpacked_elements, segment_start_index)
    
    scan_indices = jnp.arange(fitting_config.batch_size + 1, dtype=INTTYPE)
    
    final_state, collected_residuals = jax.lax.scan(
        loop_jacobian_next_iterations, 
        initial_state, 
        scan_indices
    )

    return collected_residuals.flatten().astype(DTYPE)

jac_fn_next = partial(jax.jit, static_argnames=["max_seg_len", "fitting_config","functions_config"])(jax.jacfwd(residuals_next_iterations, argnums=0),)


@partial(jax.jit, static_argnames=["max_seg_len", "fitting_config","functions_config"])
def compute_state(params_unconstrained,x_data_full, y, lower, upper, changepoint_jax,batch_index, prev_params, num_segments, leftover_batch,max_seg_len,fitting_config,functions_config):
    """
    This function transforms parameters into constrained, bounded space and computes residual, Jacobian in current state for Levenberg-Marquardt Algorithm
    
    Args:
        params_unconstrained (jnp.ndarray): Unconstrained padded JAX array of current optimized parameters
        x_data_full (jnp.ndarray): Padded JAX array of dataset x-values
        y (jnp.ndarray): Padded JAX array of dataset y-values
        lower (jnp.ndarray): Padded JAX array of lower bounds for each parameter
        upper (jnp.ndarray): Padded JAX array of upper bounds for each parameter
        changepoint_jax (jnp.ndarray): Padded JAX array of changepoint indices
        segment_lengths_jax (jnp.ndarray): Padded JAX array of segment lengths
        batch_index (jnp.int32): Current batch index
        prev_params (jnp.ndarray): JAX array of previous batch last segment's full parameters
        num_segments (jnp.int32): Number of segments to fit in this mode
        leftover_batch (jnp.int32): 1 if a partial batch exists, 0 otherwise.
        max_seg_len (int): Maximum segment length in this mode for residual array padding
        fitting_config (NamedTuple): Stores various static parameters for fitting
        functions_config (NamedTuple): Stores JAX, numpy functions and continuity settings

    Returns:
        tuple: (params_unconstrained, r, J_unconstrained, ssr)
            params_unconstrained (jnp.ndarray): Current parameters in solver space.
            r (jnp.ndarray): The current residual vector.
            J_unconstrained (jnp.ndarray): Jacobian matrix adjusted by the transform gradient.
            ssr (float): Sum of Squared Residuals (the current cost).
    """
    params = utility.to_constrained_jax(params_unconstrained, lower, upper)
    r = residuals_next_iterations(params, x_data_full,y, changepoint_jax, batch_index, prev_params, num_segments, leftover_batch,max_seg_len,fitting_config,functions_config)
    J_constrained = jac_fn_next(params,x_data_full, y, changepoint_jax, batch_index, prev_params, num_segments, leftover_batch,max_seg_len,fitting_config,functions_config)
    
    sech_squared = 1.0 / (jnp.cosh(params_unconstrained)**2)
    sech_squared = jnp.maximum(1e-12, sech_squared) # Numerical stability to prevent the gradient from vanishing
    transform_grad_vec = (upper - lower) * 0.5 * sech_squared
    
    J_unconstrained = J_constrained * transform_grad_vec
    
    ssr = jnp.dot(r,r)
    
    return params_unconstrained, r, J_unconstrained, ssr

@partial(jax.jit)
def solve_step(p, r, J, lam,ridge):
    """
    Core function for Levenberg-Marquardt algorithm and equations with numerical stability checks and clipping.

    Args:
        p (jnp.ndarray): Unconstrained padded JAX array of current optimized parameters
        r (jnp.ndarray): Padded JAX array of residuals
        J (jnp.ndarray): Padded JAX array of Jacobian values
        lam (jnp.float64): Current lambda value
        ridge (jnp.float64): Ridge used in Levenberg-Marquardt equation

    Returns:
        dp_unconstrained (jnp.ndarray): The proposed step for the parameters in unconstrained space, clipped and scaled for stability
    """
    H = J.T @ J
    g= J.T @ r
    A = H + (lam + ridge) * jnp.eye(p.size, dtype=DTYPE)
    
    bad = jnp.logical_or(jnp.any(jnp.isnan(A)), jnp.any(jnp.isinf(A)))
    diag_H = jnp.diag(H)

    fallback_scale = jnp.where(jnp.isfinite(jnp.mean(diag_H)), jnp.mean(diag_H), 1.0)
    A_fallback = (lam + ridge + fallback_scale) * jnp.eye(p.size, dtype=DTYPE)
    
    A = jnp.where(bad, A_fallback, A)
    dp_unconstrained = solve(A, -g, assume_a='pos')

    dp_unconstrained = jnp.where(jnp.isnan(dp_unconstrained), jnp.zeros_like(dp_unconstrained), dp_unconstrained)

    MAX_DP_COMPONENT = 5.0 
    dp_unconstrained = jnp.clip(dp_unconstrained, -MAX_DP_COMPONENT, MAX_DP_COMPONENT)

    MAX_DP_NORM = 0.5
    current_dp_norm = jnp.linalg.norm(dp_unconstrained)
    
    scale_factor = jnp.where(current_dp_norm > MAX_DP_NORM, 
                             MAX_DP_NORM / current_dp_norm, 
                             1.0)
    
    dp_unconstrained = dp_unconstrained * scale_factor

    return dp_unconstrained
@partial(jax.jit, static_argnames=["max_seg_len", "fitting_config","functions_config"])
def lm_fit(params_init, x_data, y, lower, upper, changepoint_jax, batch_index, prev_params,
                num_segments, leftover_batch,batch_std,batch_size, max_seg_len,fitting_config,functions_config, lam, ridge, can_converge=True,
                ftol=1e-3, xtol=1e-2):
    """
    This function transforms parameters into constrained, bounded space and computes residual, Jacobian in current state for Levenberg-Marquardt Algorithm
    
    Args:
        params_init (jnp.ndarray): Padded JAX array of initial guess parameters
        x_data (jnp.ndarray): Padded JAX array of dataset x-values
        y (jnp.ndarray): Padded JAX array of dataset y-values
        lower (jnp.ndarray): Padded JAX array of lower bounds for each parameter
        upper (jnp.ndarray): Padded JAX array of upper bounds for each parameter
        changepoint_jax (jnp.ndarray): Padded JAX array of changepoint indices
        segment_lengths_jax (jnp.ndarray): Padded JAX array of segment lengths
        batch_index (jnp.int32): Current batch index
        prev_params (jnp.ndarray): JAX array of previous batch last segment's full parameters
        num_segments (jnp.int32): Number of segments to fit in this mode
        leftover_batch (jnp.int32): 1 if a partial batch exists, 0 otherwise.
        max_seg_len (int): Maximum segment length in this mode for residual array padding
        fitting_config (NamedTuple): Stores various static parameters for fitting
        lam (jnp.float64): Initial lambda value to start optimization
        ridge (jnp.float64): Ridge used in Levenberg-Marquardt equation
        max_dp_norm (jnp.float64): Max allowed norm for proposed step
        can_converge (jnp.bool_): Boolean to allow or prevent converging to use max iterations for shaking
        ftol (jnp.float64): Relative tolerance for the change in the Sum of Squared Residuals (SSR)
        gtol (jnp.float64): Gradient norm tolerance. Stop when the norm of the gradient falls below this value, indicating a local minimum
        xtol (jnp.float64): Parameter step tolerance. Stop when the norm of the change in unconstrained parameters is smaller than xtol.

    Returns:
        tuple: (best_params, iters, best_error, converged)
            best_params (jnp.ndarray): Padded JAX array of best optimized parameters
            iters (jnp.int32): Number of iterations with which algorithm converged
            best_error (jnp.float64): Best error achieved by LM optimizer
            converged (jnp.bool_): True if any of the tolerances were met, False if max_iters was reached
    """
    lam0 = lam
    p0, r0, J0, err0 = compute_state(
        params_init, x_data,y, lower, upper, changepoint_jax,
        batch_index, prev_params, num_segments, leftover_batch, 
        max_seg_len, fitting_config,functions_config)

    init_state = (p0, lam0, err0, r0, J0, 0,jnp.inf, p0, 0, 0, False)

    def cond_fun(state):
        _, _, _, _, _, it, _, _,_,_,converged= state
        return jnp.logical_and(it < fitting_config.max_iters, jnp.logical_not(converged))

    def body_fun(state):
        p_old, lam, err_old, r_old, J_old, it, best_error, best_params, conv_ftol_counter, conv_xtol_counter, _ = state
        
        dp = solve_step(p_old, r_old, J_old, lam,ridge)
        p_prop = p_old + dp
        params_c = utility.to_constrained_jax(p_prop, lower, upper)
        r_prop = residuals_next_iterations(params_c, x_data,y, changepoint_jax,
                                           batch_index, prev_params,num_segments, 
                                           leftover_batch, max_seg_len, fitting_config,functions_config)
        err_prop = jnp.dot(r_prop,r_prop)
        ActRed = err_old - err_prop

        def accept_branch(_):
            lam_new = jnp.maximum(1e-12, lam / 3.0)
            Jc = jac_fn_next(params_c,x_data, y, changepoint_jax,
                             batch_index, prev_params,num_segments, 
                             leftover_batch, max_seg_len, fitting_config,functions_config)
            
            sech2 = 1.0 / (jnp.cosh(p_prop) ** 2)
            sech2 = jnp.maximum(1e-12, sech2)
            trans = (upper - lower) * 0.5 * sech2
            J_new = Jc * trans

            return (p_prop, lam_new, err_prop, r_prop, J_new, True)

        def reject_branch(_):
            return (p_old, lam * 10.0, err_old, r_old, J_old, False)
        converged = False        
        p_new, lam_new, err_new, r_new, J_new, is_accepted = jax.lax.cond(
            ActRed>0.0,
            accept_branch,
            reject_branch,
            operand=0
        )
        def compute_ftol():
            ftol_rel=jnp.abs(err_prop - err_old)/jnp.maximum(jnp.abs(err_old),1e-12)<ftol
            return ftol_rel  
        
        cond_ftol = jax.lax.cond(is_accepted, lambda _: compute_ftol(), lambda _: False, operand=0)
        
        def compute_xtol():
            p_step = jnp.linalg.norm(p_prop - p_old)
            p_norm = jnp.linalg.norm(p_old)
            return p_step / (p_norm + 1e-12) < xtol
        cond_xtol = jax.lax.cond(is_accepted, lambda _: compute_xtol(), lambda _: False, operand=0)
        converge_tol_count=3
        conv_ftol_counter=jnp.where(cond_ftol, conv_ftol_counter+1,0)
        conv_xtol_counter=jnp.where(cond_xtol, conv_xtol_counter+1,0)
        converged = jnp.logical_and(jnp.logical_and(jnp.logical_or(conv_ftol_counter>=converge_tol_count, conv_xtol_counter>=converge_tol_count), can_converge), it>5)

        conv_ftol_counter=jnp.where(conv_ftol_counter>=converge_tol_count, 0,conv_ftol_counter)
        conv_xtol_counter=jnp.where(conv_xtol_counter>=converge_tol_count, 0,conv_xtol_counter)
        #jax.debug.print("Iteration {it}, Error {err_new} | Lambda {lam_new} | ActRed: {ActRed}", it=it, err_new=err_new, lam_new=lam_new, ActRed=ActRed)
        best_params = jnp.where(err_new<best_error, p_new, best_params)
        best_error=jnp.where(err_new<best_error, err_new, best_error)
        return (p_new, lam_new, err_new, r_new, J_new, it + 1, best_error,best_params,conv_ftol_counter,conv_xtol_counter, converged)

    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)

    final_params, _, _, _, _, iters, best_error, best_params, _,_,converged = final_state
    return best_params, iters, best_error, converged

def forward_fit_processing(batch,batch_remainder,num_batches,num_segments,params,lower,upper,changepoint_list_original,fitting_config,functions_config):
    segments_to_fit_unpack=fitting_config.batch_size+1
    last_batch=False
    if batch==0:
        forward_fit_changepoint_array=list(changepoint_list_original[batch])
        params_to_take=(functions_config.MODEL_FULL_PARAMETER_COUNT-functions_config.MODEL_REDUCED_PARAMETER_COUNT)
    else:
        forward_fit_changepoint_array=list(np.insert(changepoint_list_original[batch], 0, changepoint_list_original[batch-1][-2]))
        params_to_take=0
    if batch==num_batches-1:
        last_batch=True
        segments_left_last_batch=0
        
        if batch_remainder==1:
            forward_fit_params_array = np.concatenate([np.array(params[batch],dtype=DTYPE), np.array(params[batch+1][:functions_config.MODEL_REDUCED_PARAMETER_COUNT],dtype=DTYPE)], axis=0)
            forward_fit_lower_array = np.concatenate([np.array(lower[batch],dtype=DTYPE), np.array(lower[batch+1][:functions_config.MODEL_REDUCED_PARAMETER_COUNT],dtype=DTYPE)], axis=0)
            forward_fit_upper_array = np.concatenate([np.array(upper[batch],dtype=DTYPE), np.array(upper[batch+1][:functions_config.MODEL_REDUCED_PARAMETER_COUNT],dtype=DTYPE)], axis=0)
            forward_fit_changepoint_array.append(changepoint_list_original[batch+1][1])
            segments_left_last_batch=fitting_config.batch_size+1
        else:                
            forward_fit_params_array=params[batch]
            forward_fit_lower_array=lower[batch]
            forward_fit_upper_array=upper[batch]
            if batch_remainder==0:
                segments_left_last_batch=fitting_config.batch_size
            else:
                segments_left_last_batch=num_segments-((num_segments//fitting_config.batch_size)*fitting_config.batch_size)
        segments_to_fit_unpack=segments_left_last_batch
    if not last_batch:
        forward_fit_params_array = np.concatenate([np.array(params[batch],dtype=DTYPE), np.array(params[batch+1][:functions_config.MODEL_REDUCED_PARAMETER_COUNT],dtype=DTYPE)], axis=0)
        forward_fit_lower_array = np.concatenate([np.array(lower[batch],dtype=DTYPE), np.array(lower[batch+1][:functions_config.MODEL_REDUCED_PARAMETER_COUNT],dtype=DTYPE)], axis=0)
        forward_fit_upper_array = np.concatenate([np.array(upper[batch],dtype=DTYPE), np.array(upper[batch+1][:functions_config.MODEL_REDUCED_PARAMETER_COUNT],dtype=DTYPE)], axis=0)
        forward_fit_changepoint_array.append(changepoint_list_original[batch+1][1])
    return last_batch,params_to_take,segments_to_fit_unpack, forward_fit_changepoint_array, forward_fit_params_array,forward_fit_lower_array,forward_fit_upper_array,params,lower,upper,changepoint_list_original

def fit_batch(last_batch,segments_to_fit_unpack,params_to_take,data_dicts,full_params_list, forward_fit_arrays, jnp_prev_params,batch_info,tols, bucketing_i,batch_std,fitting_config,functions_config):
    can_converge=True

    data_scale = batch_std
    base_ridge = 1e-4

    adaptive_ridge = max(base_ridge * (data_scale**2),1e-9)
    batch_solver,iters, best_error, conv = lm_fit(
        params_init=jnp.array(forward_fit_arrays['params'],dtype=DTYPE),
        x_data=data_dicts['x_data_jax'],
        y=data_dicts['y_data_padded_jax'],
        lower=jnp.array(forward_fit_arrays['lower'],dtype=DTYPE),
        upper=jnp.array(forward_fit_arrays['upper'],dtype=DTYPE),
        changepoint_jax=jnp.array(forward_fit_arrays['changepoint'],dtype=DTYPE),
        batch_index=jnp.array(batch_info['batch'],dtype=INTTYPE),
        prev_params=jnp_prev_params, 
        num_segments=jnp.array(batch_info['num_segments'], dtype=INTTYPE), 
        leftover_batch=jnp.array(batch_info['leftover_batch'], dtype=INTTYPE),batch_std=batch_std,batch_size=tols['batch_size'],
        max_seg_len=bucketing_i,fitting_config=fitting_config,functions_config=functions_config,lam=tols['lam'],ridge=adaptive_ridge,can_converge=jnp.array(can_converge, dtype=jnp.bool_), ftol=tols['ftol'],xtol=tols['xtol']
    )
    full_batch_solver=batch_solver
    batch_solver_constrained = utility.to_constrained_jax(
        batch_solver,
        jnp.array(forward_fit_arrays['lower'], dtype=DTYPE),
        jnp.array(forward_fit_arrays['upper'], dtype=DTYPE)
    )

    prev_params_pass=[]
    if not batch_info['batch']==0:
        prev_params_pass=full_params_list[-1][-1]

    full_params=unpack_parameters(batch_info['batch'],segments_to_fit_unpack, batch_solver_constrained, forward_fit_arrays['changepoint'], prev_params_pass, data_dicts['x_data_py'], functions_config)
    if not last_batch:
        full_params=full_params[:fitting_config.batch_size]
        batch_solver=batch_solver[:fitting_config.batch_size*functions_config.MODEL_REDUCED_PARAMETER_COUNT+params_to_take]

    return full_params, batch_solver,full_batch_solver

def lm_start(params, x_data_full_np, y_padded, lower, upper, changepoint_list_original, num_segments, segment_lengths, modes_length_bucketing,max_segment_lengths, mode, fitting_config,functions_config):
    """
    This function performs fitting for each mode by batches. Implements Forward Fit strategy, pads arrays for JAX function and uses shaking if fit is poor.
    
    Args:
        params (list[float]): list of initial guess parameters
        y_padded (list[float]): list of dataset y-values
        lower (list[float]): list of lower bounds for each parameter
        upper (list[float]): list of upper bounds for each parameter
        changepoint_list_original (list[int]): list of changepoint indices
        num_segments (int): Number of segments to fit in this mode
        segment_lengths (list[int]): list of mode segment lengths
        modes_length_bucketing (list[int]): Bucketing list of maximum segment lengths
        max_segment_lengths (list[int]): Maximum segment lengths list in all modes
        mode (int): Current mode index
        fitting_config (NamedTuple): Stores various static parameters for fitting
        functions_config (NamedTuple): Stores JAX, numpy functions and continuity settings

    Returns:
        tuple: (full_params_list, reduced_parameters)
            full_params_list (list[float]): List of full parameters for each segment
            reduced_parameters (list[float]): List of reduced parameters(without c1, c0) for each segment
    """

    full_params_list=[]
    prev_parameters=[]
    reduced_parameters=[]
    leftover_batch=1
    batch_remainder=num_segments%fitting_config.batch_size

    if (batch_remainder==0 or batch_remainder==1) and not num_segments<fitting_config.batch_size: 
        leftover_batch=0

    num_batches=max(num_segments//fitting_config.batch_size+leftover_batch,1)
    if num_segments<fitting_config.batch_size:
        batch_remainder=-1*(fitting_config.batch_size-num_segments)

    max_padding=fitting_config.batch_size+3 #past and next segments, + 1 as they are change points
    batch_size   = jnp.array(fitting_config.batch_size, dtype=jnp.int32)

    lam  = jnp.array(fitting_config.initial_lam, dtype=jnp.float64)
    ftol = jnp.array(fitting_config.ftol, dtype=jnp.float64)
    xtol = jnp.array(fitting_config.xtol, dtype=jnp.float64)

    x_data_jnp=jnp.array(x_data_full_np,dtype=DTYPE)
    y_full_jnp=jnp.array(y_padded, dtype=DTYPE)

    data_dicts={'x_data_jax': x_data_jnp, 'y_data_padded_jax': y_full_jnp,'x_data_py': x_data_full_np,'y_data_py': y_padded}
    tols={'ftol': ftol,'xtol': xtol, 'lam': lam, 'batch_size': batch_size}

    for batch in range(num_batches):
        last_batch,params_to_take,segments_to_fit_unpack, forward_fit_changepoint_array, forward_fit_params_array,forward_fit_lower_array,forward_fit_upper_array,params,lower,upper,changepoint_list_original=forward_fit_processing(batch,batch_remainder,num_batches,num_segments,params,lower,upper,changepoint_list_original,fitting_config,functions_config)

        
        if not last_batch:
            y_values = y_padded[changepoint_list_original[batch][0]:changepoint_list_original[batch+1][1]+1]
            batch_std=np.std(y_values) + 1e-12

        else:
            y_values = y_padded[changepoint_list_original[batch][0]:changepoint_list_original[batch][-1]]
            batch_std=np.std(y_values) + 1e-12 

        pad_zeros=functions_config.MODEL_FULL_PARAMETER_COUNT+functions_config.MODEL_REDUCED_PARAMETER_COUNT*fitting_config.batch_size-len(forward_fit_params_array)

        if batch==0:
            jnp_prev_params=jnp.zeros(functions_config.MODEL_FULL_PARAMETER_COUNT, dtype=DTYPE)
        else:
            jnp_prev_params=jnp.array(prev_parameters[-1],dtype=DTYPE)

        forward_fit_params_array=np.pad(forward_fit_params_array,(0, pad_zeros), mode='constant', constant_values=forward_fit_params_array[-1])
        forward_fit_lower_array=np.pad(forward_fit_lower_array,(0, pad_zeros), mode='constant', constant_values=forward_fit_lower_array[-1])
        forward_fit_upper_array=np.pad(forward_fit_upper_array,(0, pad_zeros), mode='constant', constant_values=forward_fit_upper_array[-1])
        forward_fit_changepoint_array=np.pad(forward_fit_changepoint_array,(0, max_padding-len(forward_fit_changepoint_array)), mode='constant', constant_values=forward_fit_changepoint_array[-1])
        forward_fit_arrays={'params': forward_fit_params_array, 'lower': forward_fit_lower_array, 'upper': forward_fit_upper_array, 'changepoint': forward_fit_changepoint_array}
        batch_info={'batch': batch, 'num_segments': num_segments, 'leftover_batch': leftover_batch}

        bucketing_i=0
        if fitting_config.bucketing:
            for h in range(len(modes_length_bucketing)):
                if max_segment_lengths[mode]<=modes_length_bucketing[h]:
                    bucketing_i=modes_length_bucketing[h]
                    break

        if not fitting_config.bucketing:
            bucketing_i=max_segment_lengths[mode]

        best_fit, best_batch_solver,best_full_batch_solver=fit_batch(last_batch,segments_to_fit_unpack,params_to_take,data_dicts,full_params_list, forward_fit_arrays, jnp_prev_params,batch_info,tols, bucketing_i,batch_std,fitting_config,functions_config)        
        full_params=best_fit

        batch_solver=best_batch_solver

        if not last_batch:
            if batch==0:
                params[batch+1][:functions_config.MODEL_REDUCED_PARAMETER_COUNT] = best_full_batch_solver[-functions_config.MODEL_REDUCED_PARAMETER_COUNT:]
        
            else:
                params[batch+1][:functions_config.MODEL_REDUCED_PARAMETER_COUNT]=best_full_batch_solver[len(best_full_batch_solver)-functions_config.MODEL_FULL_PARAMETER_COUNT:len(best_full_batch_solver)-(functions_config.MODEL_FULL_PARAMETER_COUNT-functions_config.MODEL_REDUCED_PARAMETER_COUNT)]
      
        prev_params=jnp.array(full_params[-1],dtype=DTYPE)
        prev_parameters.append(prev_params)
        reduced_parameters.append(batch_solver[:segments_to_fit_unpack*functions_config.MODEL_REDUCED_PARAMETER_COUNT])
        full_params_list.append(full_params[:segments_to_fit_unpack])

    return full_params_list, reduced_parameters