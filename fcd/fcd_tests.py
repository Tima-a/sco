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
def test_datasets(k, max_k=24, data_user=None, model_user=None, model_init_user=None):   
    if k<=max_k:
        data_y=np.load(f"test_datasets/cryptocoin_tests/test{k}.npy")
        if data_user:
            data_y=data_user
        x_delta_max=1.0
        if k<=5:
            x_delta_max=1.0
        elif k>5 and k<=15:
            x_delta_max=10.0
        elif k>15 and k<24:
            x_delta_max=0.01
        dataset_length=len(data_y)
        deltas = np.random.uniform(1e-4, x_delta_max, dataset_length)
        x_data = np.cumsum(deltas)
    else:
        if k == max_k+1:
            dataset_length=103
            x_data=np.arange(dataset_length)
            data_y=utility.model_linear(x_data,2.0,-100.0)
        elif k == max_k+2:
            dataset_length=987
            x_delta_max=100.0
            deltas = np.random.uniform(1e-4, x_delta_max, dataset_length)
            x_data = np.cumsum(deltas)
            data_y=utility.model_cubic(x_data,1.0, -7.0, 21.0, -23581.0)
        elif k == max_k+3:
            dataset_length=1291
            x_delta_max=-10.0
            deltas = np.random.uniform(-1e-4, x_delta_max, dataset_length)
            x_data = np.cumsum(deltas)
            data_y=np.full(dataset_length, 90.0)
        elif k == max_k+4:
            dataset_length=11
            x_delta_max=0.01
            deltas = np.random.uniform(1e-4, x_delta_max, dataset_length)
            x_data = np.cumsum(deltas)
            data_y=np.full(dataset_length, 0.0)
        elif k == max_k+5:
            data_y=np.load(f"test_datasets/cryptocoin_tests/test{1}.npy")/1e20
            x_data=np.linspace(0, 1e-16, len(data_y))
        elif k == max_k+6:
            data_y=np.load(f"test_datasets/cryptocoin_tests/test{6}.npy")/1e-20
            x_data=np.linspace(0, 1e20, len(data_y))
    if k <= max_k:
        if k>len(utility.default_models):
            k-=len(utility.default_models)
        if k == 11: # 11 is for fourier model, it is unstable for now and therefore is excluded
            k-=1
        model = list(utility.default_models.keys())[k-1]
        model_init_guesses=utility_guesses.initial_guesses_models[k-1]
    else:
        model = list(utility.default_models.keys())[5]
        model_init_guesses=utility_guesses.initial_guesses_models[5]
    if model_user:
        model=model_user
        model_init_guesses=model_init_user
    return x_data,data_y, model,model_init_guesses
 
def get_bitcoin_seed_data(seed, coin, tf, start_date):
    api_key=''
    secret_key=''
    exchange = ccxt.binance({
            'enableRateLimit': True,
            'apiKey': api_key,
            'secret': secret_key,
            'options': {'defaultType': 'spot'},
            })
    exchange.options['adjustForTimeDifference'] = True
    date_start=start_date
    limit=1000
    num_chunks=10
    since_release = int(start_date.timestamp() * 1000)
    next_since=since_release
    all_dataframes=[]

    for i in range(num_chunks):        
            ohlcv_chunk = exchange.fetch_ohlcv(f'{coin}/USDT', tf, next_since, limit=limit) 
            
            if not ohlcv_chunk:
                print(f"Fetch stopped early: No data returned for chunk {i+1} starting at {datetime.fromtimestamp(next_since / 1000)}")
                break
                
            df_chunk = pd.DataFrame(ohlcv_chunk, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_chunk['timestamp'] = pd.to_datetime(df_chunk['timestamp'], unit='ms')
    
            all_dataframes.append(df_chunk)
            last_timestamp_ms = ohlcv_chunk[-1][0]
            next_since = last_timestamp_ms + 1
            time.sleep(2)
    data_full_bitcoin = pd.concat(all_dataframes, ignore_index=True)
    open_data=(data_full_bitcoin['open'][:]).to_numpy()
    return open_data

def make_testing_dataset():
    unique_coins = ['BTC', 'ETH', 'SOL', 'ADA', 'DOGE', 'XRP']
    unique_tfs = ['1s', '1m', '1h', '1d']
    
    coins = np.repeat(unique_coins, len(unique_tfs)).tolist()
    
    tfs = np.tile(unique_tfs, len(unique_coins)).tolist()

    start_date_str = "2021-01-01 00:00:00"
    date_format = "%Y-%m-%d %H:%M:%S"
    current_date = datetime.strptime(start_date_str, date_format)
    for i in range(len(coins)):    
        dataset=get_bitcoin_seed_data(i,coins[i],tfs[i],current_date)
        safe_date_str = current_date.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'{coins[i]}_dataset_{tfs[i]}_{safe_date_str}.npy'
        np.save(filename, dataset)

def main_fitting():
    #97,11,45,76, 154, 231
    np.random.seed(97)    
    warning_segs=[]
    num_testing=30
    average_srmse_tests=[]
    average_rmse_tests=[]
    all_srmse_tests=[]
    time_tests=[]
    randomized=False
    model_user=utility.model_fourier
    model_init_user=utility_guesses.initial_guess_fourier
    for k in range(1,num_testing+1):
        settings={'show_plot': False}
        k_randomized=0
        if randomized:
            k_randomized=np.random.randint(1, num_testing)
        else:
            k_randomized=k
        mode_fitting_runner=mode_fitting.FCD(x_dataset=np.full(10000,1),y_dataset=np.full(10000,1), model=utility.model_sin7, initial_guesses_function=utility_guesses.initial_guess_sin7, settings_args=settings,parallel=True,verbose = 1)    
        x_dataset, y_dataset,model, init_guess_model=test_datasets(k_randomized,model_user=model_user,model_init_user=model_init_user) 
        print(f"Running test on seed {k}, k_randomized is {k_randomized}")
        mode_fitting_runner.set_data(x_dataset, y_dataset)
        mode_fitting_runner.set_model(model,init_guess_model)
        mode_fitting_runner.run()
        all_srmse=mode_fitting_runner.results['SRMSE']
        all_rmse=mode_fitting_runner.results['RMSE']
        all_time=mode_fitting_runner.results['Time took']
        all_data_scale=mode_fitting_runner.results['Data scale']
        dataset_std=np.std(y_dataset)
        for i in range(len(all_srmse)):
            for s in range(len(all_srmse[i])):
                last_start=1
                if s==len(all_srmse[i])-1:
                    last_start=0
                segment_deviation=np.std(y_dataset[mode_fitting_runner.all_changepoints[i][s]:mode_fitting_runner.all_changepoints[i][s+1]+last_start])
                segment_length=mode_fitting_runner.all_changepoints[i][1]-mode_fitting_runner.all_changepoints[i][0]
                threshold=2.0
                if segment_length<=15:
                    threshold=10.0
                elif segment_length<=50:
                    threshold=3.0
                if all_srmse[i][s]>threshold:
                    warning_segs.append({"test": k, "mode": i, "segment": s, "dataset deviation": f"{fmt(dataset_std)}", "segment deviation": f"{fmt(segment_deviation)}", "SRMSE": f"{fmt(all_srmse[i][s])}", "RMSE": f"{fmt(all_rmse[i][s])}"})
        all_srmse_flat = [item for sublist in all_srmse for item in sublist]
        all_rmse_flat = [item for sublist in all_rmse for item in sublist]
        all_srmse_tests.append(max(all_srmse_flat))
        if k<25: #counting only real non-stationary datasets tests
            average_srmse_tests.append(np.mean(np.array(all_srmse_flat)))
            average_rmse_tests.append(np.mean(np.array(all_rmse_flat)))
            time_tests.append(all_time)
        #print(warning_segs)
    print(f"Average SRMSE across all tests: {fmt(np.mean(average_srmse_tests))}")
    print(f"Max SRMSE across all tests: {fmt(np.max(all_srmse_tests))} at test {int(np.argmax(all_srmse_tests))+1}")
    print(f"Average RMSE across all tests: {fmt(np.mean(average_rmse_tests))}")
    print(f"Average time across all tests(with compilation): {fmt(np.mean(time_tests))}")

if __name__=="__main__":
    main_fitting()