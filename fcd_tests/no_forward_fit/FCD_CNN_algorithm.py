import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Flatten, Dense, Dropout,Input, Concatenate
import utility
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import mode_fitting
import time
import utility_guesses
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import os
from tensorflow.keras.callbacks import EarlyStopping
def set_reproducibility(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
def safe_local_scale(matrix):
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    safe_std = np.maximum(std, 1e-12)
    
    return (matrix - mean) / safe_std
def min_max_normalization(dataset):
    max_d=max(dataset)
    min_d=min(dataset)
    span=max(1e-12,max_d-min_d)
    data_scaled=(dataset-min_d)/span
    return data_scaled
def perform_mode_fitting(mode_fitting_runner, train_data, window_size):
    """
    Decomposes a window of data into physical modes and extracts features.
    This function generates the raw signal, fitted curve, and analytical derivatives (branch 1) and the optimized parameters (branch 2).
    
    Args:
        mode_fitting_runner (FCD): An instance of the FCD class initialized with the desired function
        train_data (list[float]): Array of y-values for current window
        window_size (int): size of lookback window

    Returns:
        tuple(scaled_feature_matrix,params_cnn_input)
            scaled_feature_matrix (np.ndarray): Scaled feature matrix of raw signal, fitted curve, and analytical derivatives.
            params_cnn_input (np.ndarray): Optimized parameters of each mode of decomposition for window.
    """
    x_dataset=np.arange(len(train_data))
    mode_fitting_runner.set_data(x_dataset,train_data)
    mode_fitting_runner.run()
    derivative_modes=mode_fitting_runner.calculate_derivatives()
    y_fit_modes=mode_fitting_runner.calculate_y_fit_modes()
    y_raw_col = train_data.reshape(-1, 1)
    modes_matrix = np.stack(y_fit_modes, axis=1) 
    derivs_matrix = np.stack(derivative_modes, axis=1)
    train_data_mean=np.full((window_size,1),np.mean(train_data))
    
    feature_matrix = [y_raw_col, modes_matrix,derivs_matrix]
    scaled_feature_matrix=[]
    for i in range(len(feature_matrix)):
        matrix_scaled = safe_local_scale(feature_matrix[i])
        scaled_feature_matrix.append(matrix_scaled)
    scaled_feature_matrix.append(train_data_mean)
    flat_params=[]
    change_points_normalized_modes=[]
    num_params=0
    for mode in mode_fitting_runner.fitted_parameters_modes:
        for segment in mode:
            flat_params.extend(segment)
        num_params=len(mode[0])
    for change_points_mode in mode_fitting_runner.all_changepoints:
        change_points_normalized = min_max_normalization(np.array(change_points_mode)) 
        change_points_normalized_modes.extend(np.repeat(change_points_normalized[1:], num_params).tolist()) # take only next changepoint, number of segments is len(changepoints)-1
    
    params_cnn_input = np.column_stack((flat_params, change_points_normalized_modes))
    scaled_feature_matrix=np.concatenate(scaled_feature_matrix,axis=1)
    return scaled_feature_matrix,params_cnn_input
def make_dataset_1d(target_data, window_size, prediction_size, step, mode_fitting_runner,fitted_params_modes_windows):
    """
    Generates a training windows dataset for FCD-CNN.
    
    Args:
        target_data (np.ndarray): The full scaled time-series.
        window_size (int): Lookback window size.
        prediction_size (int): Prediction length.
        step (int): The stride/step size between consecutive windows.
        mode_fitting_runner (FCD): Initialized FCD runner.
        fitted_params_modes_windows (list): Reference to a list to be populated with optimized parameters.

    Returns:
        tuple: (X, y_deltas)
            X (np.ndarray): 3D array [windows, window_size, features] for CNN input.
            y_deltas (np.ndarray): 2D array [windows, prediction_size] of CNN training.
    """
    X = []
    y_deltas = []

    for i in range(0,len(target_data) - window_size - prediction_size + 1,step):
        feature_matrix_scaled,params_cnn_input = perform_mode_fitting(mode_fitting_runner, target_data[i:i+window_size], window_size)
        fitted_params_modes_windows.append(params_cnn_input)

        X.append(feature_matrix_scaled)
        current_val = target_data[i + window_size - 1]
        future_vals = target_data[i + window_size : i + window_size + prediction_size]
        y_deltas.append(future_vals - current_val)
        
    return np.array(X), np.array(y_deltas)

def test_cnn(full_series_scaled,full_series,mode_fitting_runner, p_scaler,model, target_scaler, jump_steps ,window_size,prediction_size, training_size, num_params,num_parameters_segment,verbose):
    """
    Performs tests and evaluates predictive accuracy, calculates standard error metrics (R2, RMSE, MAE).
    
    Args:
        full_series_scaled (np.ndarray): The scaled full series.
        full_series (np.ndarray): The raw, unscaled series.
        p_scaler (StandardScaler): Scaler fitted on training parameters.
        model (Keras.Model): The trained CNN model.
        target_scaler (StandardScaler): Scaler used for the target signal.
        jump_steps (int): The stride for testing windows.
        window_size (int): Lookback window size.
        prediction_size (int): Prediction length.
        training_size (int): Number of samples used for training.
        num_params (int): Number of parameters in each window.
        num_parameters_segment (int): Number of parameters in each windows.
        verbose (int): Verbosity level.

    Returns:
        tuple: (r2, rmse, mae, prediction, true_values)
    """
    inference_start_idx = training_size + jump_steps
    inference_window,params_window = perform_mode_fitting(mode_fitting_runner,full_series_scaled[inference_start_idx : inference_start_idx + window_size],window_size)
    params_window = params_window.reshape(num_params//num_parameters_segment, num_parameters_segment,2)
    params_only=params_window[:,:,0]
    changepoints_only=params_window[:,0,1][..., np.newaxis]

    P_train=np.concatenate([params_only,changepoints_only],axis=1)
    P_reshaped = P_train.reshape(-1, P_train.shape[-1]) 
    P_scaled_flat = p_scaler.transform(P_reshaped)
    
    P_train = P_scaled_flat.reshape(P_train.shape)
    params_final = P_train[np.newaxis, ...]
    inference_input = inference_window[np.newaxis, ...]
    pred_delta_scaled = model.predict([inference_input,params_final], verbose=verbose)
    
    last_val_scaled = full_series_scaled[inference_start_idx + window_size - 1]
    final_pred_scaled = pred_delta_scaled + last_val_scaled
    prediction = target_scaler.inverse_transform(final_pred_scaled).flatten()

    true_prediction=full_series[inference_start_idx+window_size:inference_start_idx+window_size+prediction_size]
    r2_score=utility.calculate_r2(prediction,true_prediction)
    rmse_score=utility.calculate_rmse(prediction,true_prediction)
    mae_score=utility.calculate_mae(prediction,true_prediction)
    return r2_score, rmse_score, mae_score,prediction,true_prediction
def run_fcd_cnn(full_series, training_size, num_tests, verbose=0, seed=42, test_mode=0):
    """
    Executes a full training and evaluation for the FCD-enhanced CNN.
    
    Args:
        full_series (np.array): The raw time-series data.
        training_size (int): Number of samples used for training.
        num_tests (int): Number of tests to perform.
        verbose (int): Verbosity level (0 for silent, 1 for debug logs and plots, 2 for initial guess plots).
        test_mode (int): 0 for Cubic (Power data), 1 for Sin6 (EEG data).
        
    Returns:
        tuple: (average_r2, average_rmse, epochs_needed, total_time)
    """
    set_reproducibility(seed)
    start_cnn=time.perf_counter()
    target_scaler = StandardScaler()
    target_scaler.fit(full_series[:training_size].reshape(-1, 1))
    full_series_scaled = target_scaler.transform(full_series.reshape(-1, 1)).flatten()
    
    fitted_params_modes_windows=[]
    start_time_init=time.perf_counter()
    window_size = 60
    prediction_size = 30
    my_settings = {'warmup': True}

    model=utility.model_cubic
    init_guess=utility_guesses.initial_guess_cubic
    if test_mode==1:
        model=utility.model_sin6
        init_guess=utility_guesses.initial_guess_sin6
    mode_fitting_runner=mode_fitting.FCD(x_dataset=np.zeros(window_size),y_dataset=np.zeros(window_size), model=model,  initial_guesses_function=init_guess,settings_args=my_settings, parallel=True ,verbose = 0)    
    step=5
    
    X_train, y_train = make_dataset_1d(
        full_series_scaled[:training_size], 
        window_size, 
        prediction_size, step,mode_fitting_runner,fitted_params_modes_windows
    )

    if verbose>0:
        print(f"Time took FCD: {time.perf_counter()-start_time_init}")
    num_features = X_train.shape[2]
    num_windows=len(fitted_params_modes_windows)
    num_params=len(fitted_params_modes_windows[0])
    num_parameters_segment=len(mode_fitting_runner.parameter_names)
    P_train=np.array(fitted_params_modes_windows)
    P_train = P_train.reshape(num_windows,num_params//num_parameters_segment, num_parameters_segment,2)
    params_only=P_train[:,:,:,0]
    changepoints_only=P_train[:,:,0,1][..., np.newaxis]
    P_train=np.concatenate([params_only,changepoints_only],axis=2)
    P_reshaped = P_train.reshape(-1, P_train.shape[-1]) 

    p_scaler = StandardScaler()
    P_scaled_flat = p_scaler.fit_transform(P_reshaped)
    
    P_train = P_scaled_flat.reshape(P_train.shape)
    if verbose>0:
        print(f"Number of windows: {num_windows}")
        print(f"Number of features: {num_features}")

    input_cnn = Input(shape=(window_size, num_features))
    input_p = Input(shape=(num_params//num_parameters_segment, num_parameters_segment+1))
    x = Conv1D(32, 7, padding='causal', kernel_initializer='he_uniform', activation='relu')(input_cnn)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(16, 5, padding='causal', activation='relu')(x)
    x = Flatten()(x)

    y = Conv1D(32, 3, padding='causal', kernel_initializer='he_uniform', activation='relu')(input_p)
    y = BatchNormalization()(y)
    y = Conv1D(16, 3, padding='causal', kernel_initializer='he_uniform', activation='relu')(y)
    y = Flatten()(y)

    merged = Concatenate()([x, y])

    z = Dense(64, activation='relu', kernel_regularizer=l2(0.0001))(merged)
    z = BatchNormalization()(z)
    z = Dropout(0.2)(z)
    
    z = Dense(32, activation='relu', kernel_regularizer=l2(0.0001))(z)
    output = Dense(prediction_size)(z)

    
    model = Model(inputs=[input_cnn, input_p], outputs=output)
    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='mse'
    )
    early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=10,         
    restore_best_weights=True  
    )

    history=model.fit(
        [X_train, P_train], y_train,
        epochs=100,
        batch_size=16,
        shuffle=True,validation_split=0.1,callbacks=early_stop,
        verbose=verbose
    )
    epochs_needed = len(history.history['loss'])
    r2_scores=[]
    rmse_scores=[]
    mae_scores=[]
    number_of_tests=num_tests
    time_took=time.perf_counter()-start_cnn
    jump_index=20
    show_plot=False
    for i in range(number_of_tests):
        jump_steps=i*jump_index
        r2_score, rmse_score, mae_score,prediction,true_prediction=  test_cnn(full_series_scaled,full_series,mode_fitting_runner, p_scaler,model, target_scaler, jump_steps ,window_size,prediction_size, training_size, num_params,num_parameters_segment,verbose)
        if verbose>0:
            print(f"Accuracy R2 Test {i}: {r2_score}")
            print(f"Accuracy RMSE Test {i}: {rmse_score}")
            print(f"Accuracy MAE Test {i}: {mae_score}")
        r2_scores.append(r2_score)
        rmse_scores.append(rmse_score)
        mae_scores.append(mae_score)
        x_axis = np.arange(len(prediction))
        
        if verbose>0 and show_plot:
            plt.scatter(x_axis, true_prediction, color='blue', s=10, label='True')
            plt.plot(x_axis, true_prediction, color='blue', linestyle='--', alpha=0.5)
            
            plt.scatter(x_axis, prediction, color='red', s=10, label='Predicted')
            plt.plot(x_axis, prediction, color='red', linestyle='--', alpha=0.5)
            
            plt.show()
    
    if verbose>0:
        for k in range(len(r2_scores)):
            print(f"Test {k} finished with R2 of {r2_scores[k]}, RMSE of {rmse_scores[k]} and MAE of {mae_scores[k]}")
    print(f"Time took CNN: {time_took}")
    average_r2=np.mean(np.array(r2_scores))
    rmse_std=np.std(np.array(rmse_scores))
    average_rmse=np.mean(np.array(rmse_scores))
    average_mae=np.mean(np.array(mae_scores))
    if verbose>0:
        print(f"Average accuracy R2 is {average_r2}")
        print(f"Average accuracy RMSE is {average_rmse}")
        print(f"RMSE deviation is {rmse_std}")
        print(f"Average accuracy MAE is {average_mae}")
    return average_r2,average_rmse, epochs_needed,time_took

