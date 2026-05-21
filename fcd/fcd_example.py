from mode_fitting import FCD
import utility
import utility_guesses
import numpy as np

np.random.seed(112)
#Set datasets
y=np.load(f"test_datasets/cryptocoin_tests/test2.npy")[:400]
x=np.arange(len(y))
fcd = FCD(
    x_dataset=x, y_dataset=y,
    model=utility.model_sin5,
    initial_guesses_function=utility_guesses.initial_guess_sin5,
    parallel=True,
    verbose=1
)

# Execute fitting
params = fcd.run()

# Extract analytic insights
fcd.print_fitted_functions()
fitted_y_values=fcd.calculate_y_fit_modes()
derivatives = fcd.calculate_derivatives(order=1, print_derivative_formulas=True)
integrals = fcd.calculate_integrals(order=1, print_integral_formulas=True)