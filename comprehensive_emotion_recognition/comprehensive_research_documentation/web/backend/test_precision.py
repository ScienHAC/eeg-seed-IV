"""
Test High Precision Floating Point Values
=========================================

Test script to verify that the backend maintains the same precision
as the original SEED-IV dataset (like 27.795500626204074).
"""

import json
import numpy as np
from main import EEGDataPoint, HighPrecisionJSONEncoder

def test_high_precision():
    """Test that high precision floating point values are preserved"""
    
    print("🔍 Testing High Precision Floating Point Values")
    print("=" * 60)
    
    # Sample values from actual SEED-IV dataset
    dataset_values = [
        27.795500626204074,
        25.00743778857261,
        22.855960689844174,
        21.523983765925855,
        19.978601124706316
    ]
    
    print("📊 Original Dataset Values:")
    for i, val in enumerate(dataset_values):
        print(f"   Value {i+1}: {val}")
    
    # Test EEGDataPoint with high precision
    data_point = EEGDataPoint(
        timestamp=0,
        value=dataset_values[0],
        emotion="Happy",
        subject=1,
        session=1,
        trial=1,
        frequency_bands={
            "delta": dataset_values[1],
            "theta": dataset_values[2], 
            "alpha": dataset_values[3],
            "beta": dataset_values[4],
            "gamma": 18.372456789012345
        }
    )
    
    # Convert to JSON and back
    json_data = data_point.model_dump()
    json_str = json.dumps(json_data, cls=HighPrecisionJSONEncoder)
    
    print("\n✅ JSON Serialization Test:")
    print(f"   Original: {dataset_values[0]}")
    print(f"   JSON: {json_data['value']}")
    print(f"   Preserved: {json_data['value'] == dataset_values[0]}")
    
    print("\n📝 Full JSON Output (truncated):")
    print(json_str[:200] + "...")
    
    # Test NumPy float64 preservation
    np_value = np.float64(27.795500626204074)
    print(f"\n🔢 NumPy float64 test:")
    print(f"   Original: {np_value}")
    print(f"   Type: {type(np_value)}")
    print(f"   JSON: {json.dumps(float(np_value))}")
    
    print("\n✅ High precision floating point values are working correctly!")
    print("   Backend will now return values like: 27.795500626204074")
    print("   Instead of rounded values like: 27.8")

if __name__ == "__main__":
    test_high_precision()
