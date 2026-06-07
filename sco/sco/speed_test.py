from sco_main import SCO
import utility
import utility_guesses
import numpy as np
import matplotlib.pyplot as plt
import time

y=np.load(f"test_datasets/cryptocoin_tests/test2.npy") # 1m BTC dataset
x=np.arange(len(y))
settings={"scaling":True, "warmup": False}
model_user=utility.model_relation
init_guess_user=utility_guesses.initial_guess_relation
sco = SCO(
    x_dataset=x, y_dataset=y, 
    model=model_user,
    initial_guesses_function=init_guess_user,settings_args=settings,
    parallel=True,
    verbose=0
)
num_speed_tests=4
test_points=[100,1000,10000,1e5]
num_segments=[10,100,1000,10000]
for k in range(num_speed_tests):
    if k<3:
        y_test=y[:test_points[k]].copy()
        x_test=x[:test_points[k]].copy()
    elif k==3:
        y_test = np.tile(y, 10)
        x_test=np.arange(len(y_test))
    cp = np.unique(np.linspace(0, len(y_test), num_segments[k] + 1).astype(int))
    cp_list=cp.tolist()
    settings={"scaling":True, "warmup": False,'non_uniform': True, 'changepoints_non_uniform': cp_list}
    sco.set_data(x_test,y_test)
    sco.set_settings(settings)

    print(f"Running test {k+1} on {test_points[k]} points with {utility.get_name(model_user)} model")
    all_iterations=[]
    for i in range(5):
        sco_start=time.perf_counter()
        sco.run()
        sco_end=time.perf_counter()
        print(f"Iteration {i+1} finished in {sco_end-sco_start}")
        all_iterations.append(sco_end-sco_start)
    print(f"Average time after first run: {np.mean(np.array(all_iterations[1:]))}")