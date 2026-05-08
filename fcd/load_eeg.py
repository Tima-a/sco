import mne
import matplotlib.pyplot as plt
import numpy as np

file_path = "Subject00_2.edf"
raw = mne.io.read_raw_edf(file_path, preload=True)

target_channel = ['EEG C3']
data, times = raw.get_data(picks=target_channel, return_times=True)
c3_signal = data[0] * 1e6  

# 2. Basic Statistics for your paper
print(f"C3 Mean: {np.mean(c3_signal):.2f} muV")
print(f"C3 Std Dev: {np.std(c3_signal):.2f} muV")

np.save('egg_c3_signal.npy',c3_signal)
np.save('egg_c3_time.npy',times)
