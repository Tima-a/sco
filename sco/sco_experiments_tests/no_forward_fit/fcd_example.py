from mode_fitting import FCD
import utility
import utility_guesses
import numpy as np
np.random.seed(112)
#Set datasets
y=np.load(f"test_datasets/cryptocoin_tests/test3.npy")
x=np.arange(len(y))
#arr=np.array(np.random.randint(5,1000, 10)) #100 times
#arr=list(np.cumsum(arr))
#changepoints_list = []
#list_uni=[0,1000,2000,3000,4000,6000,7000,8000,9000]
#for i in range(10):
#    # 1. Randomly decide how many interior points to have (between 5 and 20)
#    # We subtract 2 because we will manually add 0 and 9999
#    num_interior_points = np.random.randint(5, 21) - 2
#    
#    # 2. Generate random unique indices between 1 and 9998
#    # We use choice to ensure no duplicates and then sort them
#    interior_points = np.random.choice(np.arange(1, 9999), size=num_interior_points, replace=False)
#    interior_points.sort()
#    
#    # 3. Combine 0, the sorted random points, and 9999
#    row = np.concatenate([[0], interior_points, [9999]])
#    
#    changepoints_list.append(list(row.astype(int)))
#settings_args={'non_uniform': True, 'changepoints_non_uniform': list_uni}
#optimization_settings_args={"batch_size": 5}
#Initialize FCD runner
fcd = FCD(
    x_dataset=x, y_dataset=y,
    model=utility.model_sin6,
    initial_guesses_function=utility_guesses.initial_guess_sin6,
    parallel=True,
    #settings_args=settings_args,
    verbose=1
)

# Execute fitting
params = fcd.run()

# Extract analytic insights
fcd.print_fitted_functions()
fitted_y_values=fcd.calculate_y_fit_modes()
derivatives = fcd.calculate_derivatives(order=1, print_derivative_formulas=True)
integrals = fcd.calculate_integrals(order=1, print_integral_formulas=True)