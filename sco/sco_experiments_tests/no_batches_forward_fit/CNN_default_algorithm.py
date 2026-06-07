import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, BatchNormalization, GlobalAveragePooling1D, Dense,Flatten,Input
import utility
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
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
    std = np.where(std < 1e-6, 1.0, std)
    
    return (matrix - mean) / std
def make_dataset(series, window_size, prediction_size, step):
    """
    Generates a training windows dataset for CNN.
    
    Args:
        series (np.ndarray): The full scaled time-series.
        window_size (int): Lookback window size.
        prediction_size (int): Prediction length.
        step (int): The stride/step size between consecutive windows.

    Returns:
        tuple: (X, predicted_x)
            X (np.ndarray): 3D array [windows, window_size, features] for CNN input.
            predicted_x (np.ndarray): 2D array [windows, prediction_size] of CNN training.
    """
    X = []
    predicted_x = []
    for i in range(0,len(series) - window_size-prediction_size+1,step):
        scaled_feature_matrix=[]
        data=series[i: i + window_size]
        data_scaled = safe_local_scale(data)[:, np.newaxis]
        data_mean=np.full((window_size,1), np.mean(data))
        scaled_feature_matrix.append(data_scaled)
        scaled_feature_matrix.append(data_mean)
        X.append(np.concatenate(scaled_feature_matrix, axis=1))
        last_known_val = series[i + window_size - 1]
        future_vals = series[i + window_size : i + window_size + prediction_size]
        predicted_x.append(future_vals - last_known_val) 
    return np.array(X), np.array(predicted_x)
def test_cnn(full_series_scaled,full_series,model, target_scaler, jump_steps ,window_size,prediction_size, training_size,verbose):
    """
    Performs tests and evaluates predictive accuracy, calculates standard error metrics (R2, RMSE, MAE).
    
    Args:
        full_series_scaled (np.ndarray): The scaled full series.
        full_series (np.ndarray): The raw, unscaled series.
        model (Keras.Model): The trained CNN model.
        target_scaler (StandardScaler): Scaler used for the target signal.
        jump_steps (int): The stride for testing windows.
        window_size (int): Lookback window size.
        prediction_size (int): Prediction length.
        training_size (int): Number of samples used for training.
        verbose (int): Verbosity level.

    Returns:
        tuple: (r2, rmse, mae, prediction, true_values)
    """
    inference_start_idx = training_size + jump_steps
    last_window = full_series_scaled[inference_start_idx:inference_start_idx+window_size]
    inf_scaled = safe_local_scale(last_window)[:, np.newaxis] 
    inf_mean = np.full((window_size, 1), np.mean(last_window))
    inf_input = np.concatenate([inf_scaled, inf_mean], axis=1)[np.newaxis, ...]
    prediction_delta_scaled = model.predict(inf_input, verbose=verbose)
    
    last_val_scaled = full_series_scaled[inference_start_idx+window_size-1]
    
    final_pred_scaled = (prediction_delta_scaled + last_val_scaled).reshape(-1, 1)
    prediction = target_scaler.inverse_transform(final_pred_scaled).flatten()
    true_prediction=full_series[inference_start_idx+window_size:inference_start_idx+window_size+prediction_size]
    r2_score=utility.calculate_r2(prediction,true_prediction)
    rmse_score=utility.calculate_rmse(prediction,true_prediction)
    mae_score=utility.calculate_mae(prediction,true_prediction)
    return r2_score, rmse_score, mae_score,prediction,true_prediction
def run_cnn(full_series, training_size, num_tests, verbose=0,seed=42):
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
    scaler = StandardScaler()
    scaler.fit(full_series[:training_size].reshape(-1, 1))
    full_series_scaled = scaler.transform(full_series.reshape(-1, 1)).flatten()
    series_scaled=full_series_scaled[:training_size]
    step=5
    
    window_size = 60
    prediction_size=30
    X, predicted_x = make_dataset(series_scaled, window_size,prediction_size, step)
    num_windows=len(X)
    num_features=len(X[0][0])
    model = Sequential([
        Input(shape=(window_size, num_features)),
        Conv1D(filters=32, kernel_size=7, padding='causal', kernel_initializer='he_uniform',activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=16, kernel_size=5, padding='causal',kernel_initializer='he_uniform', activation='relu'),        
        Flatten(),
        
        Dense(64, activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu', kernel_regularizer=l2(0.0001)),
        Dense(prediction_size)
    ])
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
        X, predicted_x,
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
        r2_score, rmse_score, mae_score,prediction,true_prediction = test_cnn(full_series_scaled,full_series,model, scaler, jump_steps ,window_size,prediction_size, training_size,verbose)
        
        if verbose>0:
            print(f"Predicted {prediction_size} next values: {prediction}")
            print(f"True prediction: {true_prediction}")
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
    
    average_r2=np.mean(np.array(r2_scores))
    average_rmse=np.mean(np.array(rmse_scores))
    average_mae=np.mean(np.array(mae_scores))
    rmse_std=np.std(np.array(rmse_scores))
    if verbose>0:
        print(f"Average accuracy R2 is {average_r2}")
        print(f"Average accuracy RMSE is {average_rmse}")
        print(f"RMSE deviation is {rmse_std}")
        print(f"Average accuracy MAE is {average_mae}")
    return average_r2,average_rmse,epochs_needed, time_took
