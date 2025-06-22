
# # Print available keys
# print(data.keys())

# # Extract EEG features and label
# eeg_features = data['data']      # shape: (62 channels, time samples)
# label = data['label'][0][0]      # e.g., 1 = positive, 2 = neutral, 3 = negative

# # Visualize EEG signal from one channel
# plt.plot(eeg_features[0])  # plot first channel
# plt.title(f"EEG Channel 1 (Label: {label})")
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.grid(True)
# plt.show()
