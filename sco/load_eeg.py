import mne
import matplotlib.pyplot as plt
import numpy as np

file_path = "test_datasets/other_tests/Subject00_2.edf"
raw = mne.io.read_raw_edf(file_path, preload=True)

target_channel = ['EEG C3']
data, times = raw.get_data(picks=target_channel, return_times=True)
print(raw.info['chs'][0]['unit'])
c3_signal = data[0] * 1e6 # to convert from volts 0.000011 V to microvolts 11 muV

# 2. Basic Statistics for your paper
print(f"C3 Mean: {np.mean(c3_signal):.2f} muV")
print(f"C3 Std Dev: {np.std(c3_signal):.2f} muV")

np.save('eeg_c3_signal.npy',c3_signal)
np.save('eeg_c3_time.npy',times)
