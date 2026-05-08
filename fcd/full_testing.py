import mode_fitting
import numpy as np
import time
import ccxt
from datetime import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta
import utility
from utility import fmt
import utility_guesses
import fcd_tests
np.random.seed(97)

def main_fitting():
    my_settings = {'scaling': True, 'warmup': False}
    optimization_settings_args={"batch_size": 5}
    mode_fitting_runner=mode_fitting.FCD(x_dataset=np.full(10000,1),y_dataset=np.full(10000,1), model=utility.model_sin7, initial_guesses_function=utility_guesses.initial_guess_sin7,settings_args=my_settings,optimization_settings_args=optimization_settings_args,parallel=True,verbose = 1)
    warning_segs=[]
    num_testing=24
    average_srmse_tests=[]
    average_rmse_tests=[]
    time_tests=[]
    model_user=utility.model_sin5
    model_init_user=utility.initial_guess_sin5
    for k in range(1,num_testing+1):
        x_dataset, y_dataset,model, init_guess_model=fcd_tests.test_datasets(k,model_user,model_init_user) 
        print(f"Running test on seed {k}")
        mode_fitting_runner.set_data(x_dataset, y_dataset)
        mode_fitting_runner.set_model(model,init_guess_model)
        mode_fitting_runner.run()

        all_srmse=mode_fitting_runner.results['SRMSE']
        all_rmse=mode_fitting_runner.results['RMSE']
        all_time=mode_fitting_runner.results['Time took']

        all_srmse_flat = [item for sublist in all_srmse for item in sublist]
        all_rmse_flat = [item for sublist in all_rmse for item in sublist]
        average_srmse_tests.append(np.mean(np.array(all_srmse_flat)))
        average_rmse_tests.append(np.mean(np.array(all_rmse_flat)))
        time_tests.append(all_time)
        #print(warning_segs)
    print(f"Average SRMSE across all tests: {fmt(np.mean(average_srmse_tests))}")
    print(f"Average RMSE across all tests: {fmt(np.mean(average_rmse_tests))}")
    print(f"Average time across all tests(with compilation): {fmt(np.mean(time_tests))}")


main_fitting()