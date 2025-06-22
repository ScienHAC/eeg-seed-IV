from scipy.io import loadmat
import matplotlib.pyplot as plt

# Load your SEED-IV eye-tracking .mat file
data = loadmat(r'C:\Users\piyus\Downloads\SEED_IV\SEED_IV\eye_feature_smooth\1\2_20150915.mat')
# data = loadmat(r'.\1_20160518.mat')

# Print the actual keys
print("Available keys:", data.keys())
print(data['eye_1'])
# Pick any available key (example: 'eye_1')
# eye_data = data['eye_1']

# # Check shape
# print("Shape of eye_1:", eye_data.shape)

# # Plot a signal (e.g., first channel/time series)
# plt.plot(eye_data[0]) 
# plt.title("Eye Feature (eye_1)")
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.grid(True)
# plt.show()
