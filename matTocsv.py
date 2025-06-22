import pandas as pd
from scipy.io import loadmat
import numpy as np
import os

def mat_to_csv(mat_file_path, csv_file_path=None):
    """
    Convert MATLAB .mat file to CSV format
    
    Args:
        mat_file_path (str): Path to the input .mat file
        csv_file_path (str): Path for the output .csv file (optional)
    """
    try:
        # Load the .mat file
        data = loadmat(mat_file_path)
        
        # Remove MATLAB metadata keys
        data = {key: value for key, value in data.items() 
                if not key.startswith('__')}
        
        # If no output path specified, create one based on input filename
        if csv_file_path is None:
            base_name = os.path.splitext(mat_file_path)[0]
            csv_file_path = f"{base_name}.csv"
        
        # Handle different data structures
        if len(data) == 1:
            # Single variable in the .mat file
            key, value = next(iter(data.items()))
            
            # Convert to pandas DataFrame
            if isinstance(value, np.ndarray):
                if value.ndim == 1:
                    df = pd.DataFrame({key: value})
                elif value.ndim == 2:
                    df = pd.DataFrame(value)
                else:
                    # For higher dimensions, flatten or handle as needed
                    df = pd.DataFrame(value.reshape(-1, value.shape[-1]))
            else:
                df = pd.DataFrame([value])
                
        else:
            # Multiple variables - combine into one DataFrame
            df_dict = {}
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    if value.ndim == 1:
                        df_dict[key] = value
                    elif value.ndim == 2 and value.shape[0] == 1:
                        df_dict[key] = value.flatten()
                    else:
                        # For complex structures, you might need custom handling
                        print(f"Warning: Variable '{key}' has complex structure, skipping")
                        continue
                else:
                    df_dict[key] = [value]
            
            df = pd.DataFrame(df_dict)
        
        # Save to CSV
        df.to_csv(csv_file_path, index=False)
        print(f"Successfully converted {mat_file_path} to {csv_file_path}")
        
        return csv_file_path
        
    except Exception as e:
        print(f"Error converting file: {e}")
        return None

def simple_mat_to_csv(mat_file, csv_file):
    """
    Simple function to convert .mat file to CSV
    
    Args:
        mat_file (str): Path to input .mat file
        csv_file (str): Path to output .csv file
    """
    # Load the .mat file
    data = loadmat(mat_file)
    print("Available keys:", list(data.keys()))
    print(list(data.values())[3])

    # Get the first data variable (skip MATLAB metadata)
    for key, value in data.items():
        if not key.startswith('__'):
            print(f"Variable name: {key}")
            print(f"Original data shape: {value.shape}")
            print(f"Data type: {type(value)}")
            
            # Count total elements using .size
            total_elements = value.size
            print(f"Total elements in dataset: {total_elements}")
            
            # For EEG data: (channels, time_samples, frequency_bands)
            # We want: time_samples as rows, channels*frequency_bands as features
            if value.ndim == 3:
                channels, time_samples, freq_bands = value.shape
                print(f"Channels: {channels}, Time samples: {time_samples}, Frequency bands: {freq_bands}")
                
                # Reshape to (time_samples, channels * freq_bands)
                # Move time dimension to first, then flatten channel and frequency dimensions
                reshaped = value.transpose(1, 0, 2)  # (time, channels, freq_bands)
                reshaped = reshaped.reshape(time_samples, channels * freq_bands)
                
                print(f"Reshaped to: {reshaped.shape} (rows=time_samples, cols=channels*freq_bands)")
                
                # Create column names
                col_names = []
                for ch in range(channels):
                    for freq in range(freq_bands):
                        col_names.append(f"Ch{ch+1}_Freq{freq+1}")
                
                df = pd.DataFrame(reshaped, columns=col_names)
                
            elif value.ndim == 2:
                # 2D array - use as is
                df = pd.DataFrame(value)
            else:
                # 1D array
                df = pd.DataFrame({key: value})
            
            # Save to CSV
            df.to_csv(csv_file, index=False)
            print(f"CSV saved with shape: {df.shape}")
            print(f"Converted {mat_file} to {csv_file}")
            break

# Manual usage examples - uncomment the one you want to use:

# Simple conversion (just change the file paths):
# simple_mat_to_csv('your_file.mat', 'output.csv')

# Advanced conversion (handles complex data structures):
# mat_to_csv('your_file.mat', 'output.csv')

# Example with actual usage:
if __name__ == "__main__":
    # Option 1: Simple conversion
    simple_mat_to_csv(r'C:\Users\piyus\Downloads\SEED_IV\SEED_IV\eeg_feature_smooth\1\1_20160518.mat', 'simple_output1.csv')
    
    # Option 2: Advanced conversion
    # mat_to_csv('filename.mat', 'advanced_output.csv')
    
    print("To use this script:")
    print("1. For simple conversion: simple_mat_to_csv('your_file.mat', 'output.csv')")
    print("2. For advanced conversion: mat_to_csv('your_file.mat', 'output.csv')")
