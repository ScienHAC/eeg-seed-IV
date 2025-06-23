import pandas as pd
from scipy.io import loadmat
import numpy as np
import os
import re

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



def extract_specific_feature(mat_file, feature_key, output_base_dir="csv"):
    """
    Extract a specific feature from .mat file and save as CSV in organized folder structure
    
    Args:
        mat_file (str): Path to input .mat file (e.g., 'C:\\Users\\...\\1\\1_20160518.mat')
        feature_key (str): Specific feature to extract (e.g., 'de_LDS1')
        output_base_dir (str): Base directory for output (default: 'csv')
    
    Returns:
        str: Path to the generated CSV file
    
    Example:
        Input: 'path/to/1/1_20160518.mat', feature='de_LDS1'
        Output: 'csv/1/1/de_LDS1.csv'
    """
    # Load the .mat file
    data = loadmat(mat_file)
    
    # Check if the feature exists
    if feature_key not in data:
        print(f"Error: Feature '{feature_key}' not found in file!")
        print(f"Available features: {[k for k in data.keys() if not k.startswith('__')]}")
        return None
    
    # Extract folder number and file identifier from path
    # Example: C:\Users\...\1\1_20160518.mat -> subject=1, session=1
    path_parts = mat_file.replace('\\', '/').split('/')
    subject_folder = path_parts[-2]  # Second to last part (subject folder)
    filename = path_parts[-1]        # Last part (filename)
    
    # Extract session identifier (first number in filename)
    file_match = re.match(r'(\d+)', filename)
    session_id = file_match.group(1) if file_match else "unknown"
    
    # Create organized folder structure: csv/subject/session/
    output_dir = os.path.join(output_base_dir, subject_folder, session_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename: feature.csv
    csv_filename = f"{feature_key}.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)    
    # Get the feature data
    value = data[feature_key]
    
    print(f"Input file: {mat_file}")
    print(f"Extracting feature: {feature_key}")
    print(f"Output directory: {output_dir}")
    print(f"Output file: {csv_filepath}")
    print(f"Original data shape: {value.shape}")
    print(f"Data type: {type(value)}")
    
    # Process the data based on its dimensions
    if value.ndim == 3:
        channels, time_samples, freq_bands = value.shape
        print(f"Channels: {channels}, Time samples: {time_samples}, Frequency bands: {freq_bands}")
        
        # Reshape to (time_samples, channels * freq_bands)
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
        df = pd.DataFrame({feature_key: value})
    
    # Save to CSV
    df.to_csv(csv_filepath, index=False)
    print(f"CSV saved with shape: {df.shape}")
    print(f"Generated file: {csv_filepath}")
    
    return csv_filepath

def simple_mat_to_csv(mat_file, csv_file):
    """
    Simple function to convert .mat file to CSV (gets first available feature)
    
    Args:
        mat_file (str): Path to input .mat file
        csv_file (str): Path to output .csv file
    """
    # Load the .mat file
    data = loadmat(mat_file)
    print("Available keys:", list(data.keys()))

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
    # Example usage for extracting a specific feature
    # Uncomment and modify the path to your .mat file
    
    # Example 1: Extract specific feature (new organized structure)
    mat_file_path = r"C:\Users\piyus\Downloads\SEED_IV\SEED_IV\eeg_feature_smooth\1\2_20150915.mat"
    feature_name = "de_movingAve1"  # or "psd_LDS1", "de_LDS1", etc.
    csv_file = extract_specific_feature(mat_file_path, feature_name)
    # Creates: csv/1/1/de_movingAve1.csv
    
    # Example 2: Extract multiple features for the same file
    # for feature in ["de_LDS1", "psd_LDS1", "de_movingAve1"]:
    #     csv_file = extract_specific_feature(mat_file_path, feature)
    #     print(f"Created: {csv_file}")
    
    # Example 3: Convert all features (original function)
    # mat_file_path = r"C:\path\to\your\file.mat"
    # csv_file = simple_mat_to_csv(mat_file_path, "output.csv")
    
    print("Ready to convert!")
    print("File structure: csv/subject/session/feature.csv")
    print("Example: csv/1/1/de_LDS1.csv")
    print("To use:")
    print("1. For specific feature: extract_specific_feature('path/to/file.mat', 'de_LDS1')")
    print("2. For all features: simple_mat_to_csv('path/to/file.mat', 'output.csv')")
