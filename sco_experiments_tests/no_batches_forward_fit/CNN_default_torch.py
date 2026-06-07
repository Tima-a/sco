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

window_size=40
prediction_size=20

class AdvancedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv1d(1, 32, 3,padding=1)
        self.bn1=nn.BatchNorm1d(32)
        self.conv2=nn.Conv1d(32,64, 3,padding=1)
        self.pool = nn.MaxPool1d(2)
        self.bn2=nn.BatchNorm1d(64)

        self.dropout=nn.Dropout(p=0.2)
        self.fc1=nn.Linear(576, 128)
        self.fc2=nn.Linear(128, prediction_size-1)

    def forward(self, x):
        # [16,19]
        x=x.unsqueeze(1) # 16 x 1 x 39
        x=self.conv1(x) #16 x 32 x 39
        x = self.pool(F.relu(self.bn1(x))) # 16 x 32 x 19
        #max pooling layer shrinks input length, never the channels
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # 16 x 64 x 9
        x=self.dropout(x)
        x = torch.flatten(x, 1) # 16 x 576
        x=F.relu(self.fc1(x)) # 16 x 128
        x = self.fc2(x) # 16 x 10
        return x

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
def generate_nn_dataset(data, window_size, prediction_size, batch_size, training):
    data_input=[]
    data_prediction=[]
    data_first_batches=[]
    validation_batches_inputs=[]
    validation_batches_predictions=[]

    for i in range(len(data)-window_size-prediction_size+1):
        data_input.append(convert_delta(data[i:i+window_size]))
        data_prediction.append(convert_delta(data[i+window_size:i+window_size+prediction_size]))
        data_first_batches.append(data[i+window_size])
    data_input_batched=torch.stack(data_input)
    data_prediction_batched=torch.stack(data_prediction)

    if training:
        val_size=int(len(data_input_batched)*0.1)
        validation_batches_inputs=data_input_batched[-val_size:]
        validation_batches_predictions=data_prediction_batched[-val_size:]
        data_input_batched=data_input_batched[:len(data_input_batched)-val_size]
        data_prediction_batched=data_prediction_batched[:len(data_prediction_batched)-val_size]

        indices = torch.randperm(data_input_batched.size(0))
        data_input_batched=data_input_batched[indices]
        data_prediction_batched=data_prediction_batched[indices]
        X_val_batches=[]
        Y_val_batches=[]
        X_batches=[]
        Y_batches=[]
        data_first=[]
        num_samples = len(validation_batches_inputs)
        for i in range(0, (num_samples // batch_size) * batch_size, batch_size):
            X_val_batches.append(validation_batches_inputs[i:i+batch_size])
            Y_val_batches.append(validation_batches_predictions[i:i+batch_size])

        num_samples = data_input_batched.size(0)
        for i in range(0, (num_samples // batch_size) * batch_size, batch_size):
            X_batches.append(data_input_batched[i:i+batch_size])
            Y_batches.append(data_prediction_batched[i:i+batch_size])
            data_first.append(data_first_batches[i:i+batch_size])
    else:
        X_batches=[]
        Y_batches=[]
        data_first=[]
        num_samples = data_input_batched.size(0)
        for i in range(0, (num_samples // batch_size) * batch_size, batch_size):
            X_batches.append(data_input_batched[i:i+batch_size])
            Y_batches.append(data_prediction_batched[i:i+batch_size])
            data_first.append(data_first_batches[i:i+batch_size])
        return X_batches, Y_batches,data_first
    return X_batches,Y_batches, data_first, X_val_batches, Y_val_batches


def mse(predictions, actual):
    return np.mean((predictions-actual)**2)
def main():
    data_train_original = np.load("./test_datasets/other_tests/UCI_household_power_dataset.npy")[:9000]
    data_test_original = np.load("./test_datasets/other_tests/UCI_household_power_dataset.npy")[9000:10000]


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
        data_input, data_prediction,_, data_val_input, data_val_prediction=generate_nn_dataset(data_train, window_size, prediction_size, batch_size, True)
        data_input,data_prediction=torch.stack(data_input).to(device), torch.stack(data_prediction).to(device)
        data_val_input,data_val_prediction=torch.stack(data_val_input).to(device), torch.stack(data_val_prediction).to(device)
        loss_epochs=[]

        for i in range(len(data_input)-1):
            inputs, predictions = data_input[i], data_prediction[i]

            optimizer.zero_grad()
            outputs = net(inputs)
            loss=criterion(outputs, predictions)
            loss.backward()
            optimizer.step()

            loss_epochs.append(loss.item())

        #Validation error is calculated each epoch
        with torch.no_grad():
            loss_validation_arr=[]

            for z in range(len(data_val_input)):
                inputs_validation, predictions_validation = data_val_input[z], data_val_prediction[z]
                outputs_validation = net(inputs_validation)
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
    data_input, data_prediction,data_first_batches=generate_nn_dataset(data_test, window_size, prediction_size,batch_size,False)
    data_input,data_prediction=torch.stack(data_input).to(device), torch.stack(data_prediction).to(device)
    all_mse=[]

    #Testing
    with torch.no_grad():
        for i in range(len(data_input)):
            inputs, predictions = data_input[i], data_prediction[i]
            outputs = net(inputs)

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