from mode_fitting import FCD
import utility
import utility_guesses
import numpy as np
import matplotlib.pyplot as plt
import time

y=np.load(f"test_datasets/cryptocoin_tests/test2.npy") # 1m BTC dataset
x=np.arange(len(y))
settings={"scaling":True, "warmup": False}
fcd = FCD(
    x_dataset=x, y_dataset=y, 
    model=utility.model_cubic,
    initial_guesses_function=utility_guesses.initial_guess_cubic,settings_args=settings,
    parallel=True,
    verbose=0
)
num_speed_tests=5
test_points=[10,100,1000,10000,1e5]
for k in range(num_speed_tests):
    if k<4:
        y_test=y[:test_points[k]].copy()
        x_test=x[:test_points[k]].copy()
    elif k==4:
        y_test = np.tile(y, 10)
        x_test=np.arange(len(y_test))
    fcd.set_data(x_test,y_test)
    print(f"Running test {k+1} on {test_points[k]} points with cubic model")
    for i in range(5):
        fcd_start=time.perf_counter()
        fcd.run()
        fcd_end=time.perf_counter()
        print(f"Iteration {i+1} finished in {fcd_end-fcd_start}")