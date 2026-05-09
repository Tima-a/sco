import CNN_default_algorithm
import FCD_CNN_algorithm 
import numpy as np

full_series=None
test_mode=0
if test_mode==0:
    full_series = np.load("test_datasets/other_tests/UCI_household_power_dataset.npy")
elif test_mode==1:
    full_series = (np.loadtxt('test_datasets/other_tests/s00.csv', usecols=1, delimiter=','))
training_sizes = [5000, 10000, 20000]
num_tests = [50, 50, 50]
seeds=[3,134,49, 96, 18]
for i in range(0,len(training_sizes)):
    tests_cnn = []
    tests_fcd_cnn = []
    print(f"Testing on training size of {training_sizes[i]}")
    for s in range(5):
        print(f"Iteration {s}")
        r2_std,rmse_std, epochs_took,t_std = CNN_default_algorithm.run_cnn(full_series, training_sizes[i], num_tests[i],0, seeds[s])
        tests_cnn.append([r2_std, rmse_std,epochs_took, t_std])

        r2_fcd, rmse_fcd, epochs_took_fcd,t_fcd = FCD_CNN_algorithm.run_fcd_cnn(full_series, training_sizes[i], num_tests[i], 0,seeds[s], test_mode=test_mode)
        tests_fcd_cnn.append([r2_fcd, rmse_fcd, epochs_took_fcd, t_fcd])
        print(f"FCD CNN results: RMSE is {rmse_fcd}, R2 is {r2_fcd}, Time took {t_fcd}")
        print(f"Standard CNN results: RMSE is {rmse_std}, R2 is {r2_std}, Time took {t_std}")
        print(f"FCD CNN took {epochs_took_fcd}")
        print(f"Standard CNN took {epochs_took}")
    res_cnn = np.array(tests_cnn)
    res_fcd = np.array(tests_fcd_cnn)

    print(f"\nTest Results:")
    print(f"FCD-CNN:      RMSE {np.mean(res_fcd[:, 1]):.4f} (+/- {np.std(res_fcd[:, 1]):.4f}), R2 {np.mean(res_fcd[:, 0]):.4f}, Average Epochs: {np.mean(res_fcd[:, 2]):.4f}")
    print(f"Standard CNN: RMSE {np.mean(res_cnn[:, 1]):.4f} (+/- {np.std(res_cnn[:, 1]):.4f}), R2 {np.mean(res_cnn[:, 0]):.4f}, Average Epochs: {np.mean(res_cnn[:, 2]):.4f}")
