#!/usr/bin/env python3
"""
Example script demonstrating MATLAB to CSV conversion for SEED-IV dataset
"""

from matTocsv import extract_specific_feature, simple_mat_to_csv

def main():
    """
    Example usage of the MATLAB conversion functions
    """
    print("SEED-IV Dataset Conversion Examples")
    print("=" * 50)
    
    # Example file paths (update these to match your data location)
    example_files = [
        r"C:\path\to\SEED_IV\1\1_20160518.mat",
        r"C:\path\to\SEED_IV\2\2_20160725.mat",
        r"C:\path\to\SEED_IV\3\3_20160413.mat"
    ]
    
    # Available features in SEED-IV dataset
    available_features = [
        "de_LDS1",          # Differential Entropy with Linear Dynamic System
        "psd_LDS1",         # Power Spectral Density with LDS
        "de_movingAve1",    # Differential Entropy with Moving Average
        "psd_movingAve1"    # Power Spectral Density with Moving Average
    ]
    
    print("\n1. EXTRACT SPECIFIC FEATURE (Recommended)")
    print("-" * 40)
    print("This method extracts one specific feature and creates")
    print("a CSV file with an auto-generated filename.\n")
    
    for feature in available_features:
        print(f"Feature: {feature}")
        print(f"  Usage: extract_specific_feature('path/to/1/1_20160518.mat', '{feature}')")
        print(f"  Output: 1_1_{feature}.csv")
        print()
    
    print("\n2. CONVERT ALL FEATURES")
    print("-" * 40)
    print("This method converts the first available feature in the file.\n")
    print("Usage: simple_mat_to_csv('input.mat', 'output.csv')")
    
    print("\n3. ACTUAL CONVERSION EXAMPLE")
    print("-" * 40)
    print("To actually convert files, uncomment the lines below:")
    print()
    
    # Uncomment these lines to perform actual conversions:
    
    # Example 1: Extract specific feature
    # mat_file = r"C:\path\to\your\data\1\1_20160518.mat"
    # csv_file = extract_specific_feature(mat_file, "de_LDS1")
    # print(f"Created: {csv_file}")
    
    # Example 2: Batch conversion of multiple features
    # for feature in ["de_LDS1", "psd_LDS1"]:
    #     csv_file = extract_specific_feature(mat_file, feature)
    #     print(f"Created: {csv_file}")
    
    print("\n4. FILE PATH EXAMPLES")
    print("-" * 40)
    print("Update the file paths to match your data structure:")
    for i, example_file in enumerate(example_files, 1):
        print(f"Subject {i}: {example_file}")
    
    print("\n5. GETTING STARTED")
    print("-" * 40)
    print("Steps to convert your data:")
    print("1. Update the file paths in this script")
    print("2. Uncomment the conversion lines")
    print("3. Run: python example_conversion.py")
    print("4. Check for generated CSV files in the same directory")

if __name__ == "__main__":
    main()
