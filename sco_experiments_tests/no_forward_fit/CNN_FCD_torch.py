import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
import wandb
import os
import json
import utility
import mode_fitting
import utility_guesses
window_size=40
prediction_size=20

class AdvancedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1_conv1=nn.Conv1d(6, 32, 3,padding=1)
        self.b1_bn1=nn.BatchNorm1d(32)
        self.b1_conv2=nn.Conv1d(32,64, 3,padding=1)
        self.pool = nn.MaxPool1d(2)
        self.b1_bn2=nn.BatchNorm1d(64)

        self.b1_dropout=nn.Dropout(p=0.2)

        self.b2_conv1=nn.Conv1d(2, 16, 3,padding=1)
        self.b2_bn1=nn.BatchNorm1d(16)
        self.b2_fc=nn.Linear(16*16, 64)

        self.dropout=nn.Dropout(p=0.2)
        self.fc1=nn.Linear(576+64, 128)
        self.fc2=nn.Linear(128, prediction_size-1)

    def forward(self, x1,x2):
        # x1 is [16,39, 6] and x2 is [16,32,2]
        x1=x1.transpose(1,2) # 16 x 1 x 39
        x1 = self.pool(F.relu(self.b1_bn1(self.b1_conv1(x1))))
        x1 = self.pool(F.relu(self.b1_bn2(self.b1_conv2(x1))))
        x1 = torch.flatten(x1, 1)
        #max pooling layer shrinks input length, never the channels
        x2 = x2.transpose(1, 2)
        x2 = self.pool(F.relu(self.b2_bn1(self.b2_conv1(x2))))
        x2 = torch.flatten(x2, 1)
        x2 = F.relu(self.b2_fc(x2))

        merged = torch.cat((x1, x2), dim=1)
        x = self.dropout(merged)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
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
def convert_delta(data):
    data_change=[]
    for i in range(len(data)-1):
        data_change.append(data[i+1]-data[i])
    return torch.tensor(data_change).float() # size is less by 1
def convert_undelta(data, first_value):
    data_change=[first_value]
    for i in range(len(data)):
        data_change.append(data_change[i]+data[i])
    return np.array(data_change) # size is more by 1
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
    train_data_mean=np.full((window_size-1,1),np.mean(train_data))
    
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
def generate_nn_dataset(data, window_size, prediction_size, batch_size, training,mode_fitting_runner):
    data_input=[]
    data_prediction=[]
    data_first_batches=[]
    validation_batches_inputs=[]
    validation_batches_predictions=[]
    data_params=[]
    for i in range(len(data)-window_size-prediction_size+1):
        data_batch=convert_delta(data[i:i+window_size])
        feature_matrix_scaled,params_cnn_input = perform_mode_fitting(mode_fitting_runner, data_batch.numpy(), window_size)
        data_input.append(torch.from_numpy(feature_matrix_scaled).float())
        data_params.append(torch.from_numpy(params_cnn_input).float())
        data_prediction.append(convert_delta(data[i+window_size:i+window_size+prediction_size]))
        data_first_batches.append(data[i+window_size])
    data_input_batched=torch.stack(data_input)
    data_params_batched=torch.stack(data_params)
    data_prediction_batched=torch.stack(data_prediction)

    if training:
        val_size=int(len(data_input_batched)*0.1)
        validation_batches_inputs=data_input_batched[-val_size:]
        validation_batches_params=data_params_batched[-val_size:]
        validation_batches_predictions=data_prediction_batched[-val_size:]
        data_input_batched=data_input_batched[:len(data_input_batched)-val_size]
        data_prediction_batched=data_prediction_batched[:len(data_prediction_batched)-val_size]
        data_params_batched=data_params_batched[:len(data_params_batched)-val_size]

        indices = torch.randperm(data_input_batched.size(0))
        data_input_batched=data_input_batched[indices]
        data_params_batched=data_params_batched[indices]
        data_prediction_batched=data_prediction_batched[indices]
        X_val_batches_input=[]
        X_val_batches_params=[]

        Y_val_batches=[]
        X_batches_input=[]
        X_batches_params=[]
        Y_batches=[]
        data_first=[]
        num_samples = len(validation_batches_inputs)
        for i in range(0, (num_samples // batch_size) * batch_size, batch_size):
            X_val_batches_input.append(validation_batches_inputs[i:i+batch_size])
            X_val_batches_params.append(validation_batches_params[i:i+batch_size])
            Y_val_batches.append(validation_batches_predictions[i:i+batch_size])

        num_samples = data_input_batched.size(0)
        for i in range(0, (num_samples // batch_size) * batch_size, batch_size):
            X_batches_input.append(data_input_batched[i:i+batch_size])
            X_batches_params.append(data_params_batched[i:i+batch_size])
            Y_batches.append(data_prediction_batched[i:i+batch_size])
            data_first.append(data_first_batches[i:i+batch_size])
    else:
        X_batches_input=[]
        X_batches_params=[]
        Y_batches=[]
        data_first=[]
        num_samples = data_input_batched.size(0)
        for i in range(0, (num_samples // batch_size) * batch_size, batch_size):
            X_batches_input.append(data_input_batched[i:i+batch_size])
            X_batches_params.append(data_params_batched[i:i+batch_size])
            Y_batches.append(data_prediction_batched[i:i+batch_size])
            data_first.append(data_first_batches[i:i+batch_size])
        return X_batches_input,X_batches_params, Y_batches,data_first
    return X_batches_input,X_batches_params,Y_batches, data_first, X_val_batches_input, X_val_batches_params, Y_val_batches


def mse(predictions, actual):
    return np.mean((predictions-actual)**2)
def main():
    data_train_original = np.load("./test_datasets/other_tests/UCI_household_power_dataset.npy")[:9000]
    data_test_original = np.load("./test_datasets/other_tests/UCI_household_power_dataset.npy")[9000:10000]

    my_settings = {'warmup': True}
    test_mode=0
    model=utility.model_cubic
    init_guess=utility_guesses.initial_guess_cubic
    if test_mode==1:
        model=utility.model_sin6
        init_guess=utility_guesses.initial_guess_sin6
    mode_fitting_runner=mode_fitting.FCD(x_dataset=np.zeros(window_size),y_dataset=np.zeros(window_size), model=model,  initial_guesses_function=init_guess,settings_args=my_settings, parallel=True ,verbose = 0)    
    
    # Standard scaling
    data_train=((data_train_original - np.mean(data_train_original))/np.std(data_train_original)).astype(np.float32)
    data_test=((data_test_original - np.mean(data_train_original))/np.std(data_train_original)).astype(np.float32)

    device=torch.device("cuda")
    net=AdvancedCNN().to(device)
    learning_rate=0.001
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    batch_size=16
    epochs=100

    print("Started training")
    time_start=time.perf_counter()
    all_loss_validations=[]
    min_loss_val=np.inf
    epochs_early_stop=0
    epochs_to_stop=20
    best_model_state=None

    #Training
    for epoch in range(epochs):
        #Generating the dataset of [windows, actual predictions] and conerting them to GPU
        data_input, data_params, data_prediction,_, data_val_input, data_val_params, data_val_prediction=generate_nn_dataset(data_train, window_size, prediction_size, batch_size, True,mode_fitting_runner)
        data_input,data_params,data_prediction=torch.stack(data_input).to(device), torch.stack(data_params).to(device), torch.stack(data_prediction).to(device)
        data_val_input,data_val_params,data_val_prediction=torch.stack(data_val_input).to(device),torch.stack(data_val_params).to(device), torch.stack(data_val_prediction).to(device)
        loss_epochs=[]

        for i in range(len(data_input)-1):
            inputs, inputs_params, predictions = data_input[i], data_params[i], data_prediction[i]

            optimizer.zero_grad()
            outputs = net(inputs, inputs_params)
            loss=criterion(outputs, predictions)
            loss.backward()
            optimizer.step()

            loss_epochs.append(loss.item())

        #Validation error is calculated each epoch
        with torch.no_grad():
            loss_validation_arr=[]

            for z in range(len(data_val_input)):
                inputs_validation, val_params, predictions_validation = data_val_input[z],data_val_params[z], data_val_prediction[z]
                outputs_validation = net(inputs_validation,val_params)
                loss_validation=criterion(outputs_validation,predictions_validation)
                loss_val=loss_validation.detach().cpu().numpy()
                loss_validation_arr.append(loss_validation.detach().cpu().numpy())

            loss_validation=np.mean(np.array(loss_validation_arr))
            all_loss_validations.append(loss_validation)
        loss_train_epochs=np.mean(np.array(loss_epochs))


        #Early stopping and saving best model
        if loss_validation < min_loss_val:
            min_loss_val=loss_validation
            epochs_early_stop=0
            best_model_state = net.state_dict()
        else:
            epochs_early_stop+=1
        if epochs_early_stop>=epochs_to_stop:
            print(f"Early stopping on epoch {epoch}, val_loss is {loss_validation}")
            net.load_state_dict(best_model_state)
            break
        print(f"Epoch {epoch}, Average Loss: {loss_train_epochs}, Loss validation: {loss_validation}")

    print("Finished training")
    time_took = time.perf_counter()-time_start

    #Validation error plot
    plt.figure(figsize=(12,5))
    plt.plot(all_loss_validations, color='green')
    plt.title("Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
    loss=0.0

    r2_scores=[]
    all_outputs=[]
    all_actuals=[]
    net.eval()
    data_input, data_params, data_prediction,data_first_batches=generate_nn_dataset(data_test, window_size, prediction_size,batch_size,False,mode_fitting_runner)
    data_input,data_params,data_prediction=torch.stack(data_input).to(device), torch.stack(data_params).to(device),torch.stack(data_prediction).to(device)
    all_mse=[]

    #Testing
    with torch.no_grad():
        for i in range(len(data_input)):
            inputs, params, predictions = data_input[i],data_params[i], data_prediction[i]
            outputs = net(inputs,params)

            out_arr = outputs.detach().cpu().numpy()
            pred_arr = predictions.detach().cpu().numpy()
            
            for z in range(len(out_arr)):
                all_outputs.append(convert_undelta(out_arr[z], data_first_batches[i][z]))
                all_actuals.append(convert_undelta(pred_arr[z], data_first_batches[i][z]))
                error_mse=mse(all_outputs[-1],all_actuals[-1])
                all_mse.append(error_mse)
                print(f"Test {i}, MSE error is {error_mse}")

    print(f"Average MSE error is {np.mean(np.array(all_mse))}")
    print(f"Time took: {time_took}")

    #Showing CNN prediction vs. actual prediction on graph
    for i in range(min(len(all_outputs),10)):
        plt.figure(figsize=(12, 5))
        plt.plot(all_outputs[i], color='blue', label='Predicted')
        plt.plot(all_actuals[i], color='red', label='Actual', linestyle='--')
        plt.legend()
        plt.title("Full Prediction Timeline")
        plt.show()

if __name__=='__main__':
    main()