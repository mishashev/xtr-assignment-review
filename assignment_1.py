import numpy as np
import pandas as pd
import math
from scipy import signal
import matplotlib.pyplot as plt


# using class as required,define all the prameters
class ChannelDataConverter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.num_channels = 8
        self.num_ADC_bits = 16
        self.voltage_resolution = 4.12 * (math.e ** -7)
        self.samples_per_channel = 4000

    # part of the code that convert the data
    def convert_to_dataframes(self):
        total_samples = self.num_channels * self.samples_per_channel
        data = np.fromfile(self.file_path, dtype=np.uint16, count=total_samples)
        data = np.reshape(data, (self.num_channels, -1), order='F')

        converted_data = np.multiply(self.voltage_resolution,
                                     (data - np.float_power(2, self.num_ADC_bits - 1)))
        # array for each channel
        dfs_per_channel = []
        for channel_data in converted_data:
            df = pd.DataFrame({'Amplitude': channel_data})
            dfs_per_channel.append(df)

        return dfs_per_channel

    # the bandpass filter as required
    def apply_bandpass_filter(self, channel_data):
        fs = 4000.0
        lowcut = 50
        highcut = 155
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_channel_data = signal.lfilter(b, a, channel_data)
        return filtered_channel_data

    # name of the file that hold the data

if __name__ == '__main__':
    file_path = r"C:\Users\misha\PycharmProjects\api_dataset_processor\NEUR0000.DT8"
    converter = ChannelDataConverter(file_path)
    channel_dataframes = converter.convert_to_dataframes()

    for i, df in enumerate(channel_dataframes):
        print(f"Channel {i + 1} DataFrame:")
        print(df)

        filtered_data = converter.apply_bandpass_filter(df['Amplitude'].values)
        # plot that shows the data
        plt.figure(figsize=(12, 8))
        plt.plot(df['Amplitude'].values, label='Original Data')
        plt.plot(filtered_data, label='Filtered Data')
        plt.title(f"Channel {i + 1} Data with Bandpass Filter")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.show()

