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
        mat_file (str): Path to input .mat file (e.g., 'C:\\\\Users\\\\...\\\\1\\\\1_20160518.mat')
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
    # Example: C:\\Users\\...\\1\\1_20160518.mat -> subject=1, session=1
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

def batch_convert_seed_iv(base_path, output_base_dir="csv"):
    """
    Automated batch conversion of all SEED-IV .mat files to CSV
    Processes all subjects, sessions, and extracts de_LDS and de_movingAve features
    
    Args:
        base_path (str): Path to SEED_IV/eeg_feature_smooth directory
        output_base_dir (str): Base directory for CSV output (default: 'csv')
      Example:
        base_path = r"C:\\Users\\piyus\\Downloads\\SEED_IV\\SEED_IV\\eeg_feature_smooth"
        batch_convert_seed_iv(base_path)
    """
    import glob
    
    # Features to extract (de_LDS and de_movingAve with all trial numbers)
    target_features = []
    
    # Add de_LDS features (trials 1-24 typically)
    for trial in range(1, 25):
        target_features.append(f"de_LDS{trial}")
    
    # Add de_movingAve features (trials 1-24 typically)
    for trial in range(1, 25):
        target_features.append(f"de_movingAve{trial}")
    
    print("=" * 60)
    print("SEED-IV AUTOMATED BATCH CONVERSION")
    print("=" * 60)
    print(f"Base path: {base_path}")
    print(f"Target features: de_LDS1-24, de_movingAve1-24")
    print(f"Output directory: {output_base_dir}")
    
    # Find all .mat files in the directory structure
    mat_pattern = os.path.join(base_path, "*", "*.mat")
    mat_files = glob.glob(mat_pattern)
    
    if not mat_files:
        print(f"No .mat files found in {base_path}")
        print("Please check the path and make sure it points to eeg_feature_smooth directory")
        return
    
    print(f"\nFound {len(mat_files)} .mat files to process")
    
    successful_conversions = 0
    failed_conversions = 0
    total_csv_files = 0
    
    for mat_file in mat_files:
        print(f"\n{'='*50}")
        print(f"Processing: {os.path.basename(mat_file)}")
        print(f"Full path: {mat_file}")
        
        try:
            # Load the file first to see what features are available
            from scipy.io import loadmat
            data = loadmat(mat_file)
            available_features = [k for k in data.keys() if not k.startswith('__')]
            
            print(f"Available features: {available_features}")
            
            # Extract each target feature that exists in this file
            file_csv_count = 0
            for feature in target_features:
                if feature in available_features:
                    try:
                        csv_file = extract_specific_feature(mat_file, feature, output_base_dir)
                        if csv_file:
                            file_csv_count += 1
                            total_csv_files += 1
                            print(f"✓ Created: {csv_file}")
                    except Exception as e:
                        print(f"✗ Failed to extract {feature}: {e}")
                        
            print(f"Extracted {file_csv_count} features from this file")
            successful_conversions += 1
            
        except Exception as e:
            print(f"✗ Failed to process {mat_file}: {e}")
            failed_conversions += 1
    
    # Summary report
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    print(f"Total .mat files found: {len(mat_files)}")
    print(f"Successfully processed: {successful_conversions}")
    print(f"Failed to process: {failed_conversions}")
    print(f"Total CSV files created: {total_csv_files}")
    print(f"Output directory: {os.path.abspath(output_base_dir)}")
    
    return {
        'total_files': len(mat_files),
        'successful': successful_conversions,
        'failed': failed_conversions,
        'csv_created': total_csv_files
    }

def quick_convert_all_features(base_path, output_base_dir="csv"):
    """
    Quick conversion that automatically detects and converts all de_LDS and de_movingAve features
    
    Args:
        base_path (str): Path to SEED_IV/eeg_feature_smooth directory
        output_base_dir (str): Base directory for CSV output
    """
    import glob
    
    print("=" * 60)
    print("QUICK CONVERSION - AUTO-DETECT FEATURES")
    print("=" * 60)
    
    # Find all .mat files
    mat_pattern = os.path.join(base_path, "*", "*.mat")
    mat_files = glob.glob(mat_pattern)
    
    if not mat_files:
        print(f"No .mat files found in {base_path}")
        return
    
    print(f"Found {len(mat_files)} .mat files")
    
    total_csv_files = 0
    
    for mat_file in mat_files:
        print(f"\nProcessing: {os.path.basename(mat_file)}")
        
        try:
            # Load file and find de_LDS and de_movingAve features
            from scipy.io import loadmat
            data = loadmat(mat_file)
            available_features = [k for k in data.keys() if not k.startswith('__')]
            
            # Filter for de_LDS and de_movingAve features
            target_features = [f for f in available_features 
                             if f.startswith('de_LDS') or f.startswith('de_movingAve')]
            
            print(f"Found {len(target_features)} target features: {target_features}")
            
            # Extract each feature
            for feature in target_features:
                try:
                    csv_file = extract_specific_feature(mat_file, feature, output_base_dir)
                    if csv_file:
                        total_csv_files += 1
                        print(f"✓ {feature} -> {os.path.basename(csv_file)}")
                except Exception as e:
                    print(f"✗ Failed {feature}: {e}")
                    
        except Exception as e:
            print(f"✗ Failed to process {mat_file}: {e}")
    
    print(f"\nTotal CSV files created: {total_csv_files}")
    return total_csv_files

# Example with actual usage:
if __name__ == "__main__":
    # AUTOMATED BATCH CONVERSION - RECOMMENDED FOR PROCESSING ALL FILES
    # Update this path to point to your SEED_IV/eeg_feature_smooth directory
    seed_iv_base_path = r"C:\Users\piyus\Downloads\SEED_IV\SEED_IV\eeg_feature_smooth"
    
    print("SEED-IV Automated Conversion System")
    print("==================================")
    print("Choose conversion method:")
    print("1. Quick conversion (auto-detect de_LDS and de_movingAve features)")
    print("2. Full conversion (all trials 1-24 for de_LDS and de_movingAve)")
    print("3. Single file conversion (manual)")
    
    # Method 1: Quick conversion (RECOMMENDED)
    print("\n" + "="*50)
    print("METHOD 1: QUICK CONVERSION (RECOMMENDED)")
    print("="*50)
    results = quick_convert_all_features(seed_iv_base_path)
    
    # Method 2: Full batch conversion (uncomment if you want all trials 1-24)
    # print("\n" + "="*50)
    # print("METHOD 2: FULL BATCH CONVERSION")
    # print("="*50)
    # results = batch_convert_seed_iv(seed_iv_base_path)
    
    # Method 3: Single file example (for testing individual files)
    # print("\n" + "="*50)
    # print("METHOD 3: SINGLE FILE EXAMPLE")
    # print("="*50)
    # mat_file_path = r"C:\Users\piyus\Downloads\SEED_IV\SEED_IV\eeg_feature_smooth\2\2_20150920.mat"
    # for feature in ["de_LDS1", "de_movingAve1", "de_LDS2", "de_movingAve2"]:
    #     csv_file = extract_specific_feature(mat_file_path, feature)
    #     if csv_file:
    #         print(f"Created: {csv_file}")
    
    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS:")
    print("="*60)
    print("✓ CSV files are organized in: csv/subject/session/feature.csv")
    print("✓ Example output: csv/1/1/de_LDS1.csv, csv/1/1/de_movingAve1.csv")
    print("✓ Features extracted: de_LDS* and de_movingAve* (all available trials)")
    print("✓ To change the base path, edit 'seed_iv_base_path' variable above")
    print("✓ To process specific features only, use extract_specific_feature()")
    
    print("\nFile structure created: csv/subject/session/feature.csv")
    print("Example: csv/1/1/de_LDS1.csv, csv/2/1/de_movingAve3.csv")
    
    # To run batch conversion manually, uncomment one of the following:
    # Method 2: Full batch conversion (processes trials 1-24 for all features)
    # results = batch_convert_seed_iv(seed_iv_base_path)
    # print(f"Batch conversion results: {results}")
