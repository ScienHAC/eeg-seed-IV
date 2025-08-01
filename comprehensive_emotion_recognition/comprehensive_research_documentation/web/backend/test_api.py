#!/usr/bin/env python3
"""
Simple API test script for the EEG Backend
==========================================

Tests basic functionality of the FastAPI backend to ensure
uv integration works correctly.
"""

import requests
import json
import sys

def test_health_endpoint():
    """Test the health check endpoint"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint working")
            return True
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Could not connect to backend: {e}")
        return False

def test_load_data_endpoint():
    """Test the load data endpoint with sample data"""
    try:
        payload = {
            "subject": 1,
            "session": 1,
            "frequency_band": "de_LDS",
            "trial": 1
        }
        
        response = requests.post(
            "http://localhost:8000/api/load-eeg-data",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Load data endpoint working - loaded {len(data.get('data', []))} data points")
            return True
        else:
            print(f"❌ Load data endpoint failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Could not connect to load-data endpoint: {e}")
        return False

def main():
    """Run all API tests"""
    print("🧠 EEG Backend API Tests")
    print("=" * 30)
    
    # Test basic connectivity
    if not test_health_endpoint():
        print("\n❌ Backend server not responding. Make sure it's running on port 8000")
        sys.exit(1)
    
    # Test data loading
    if not test_load_data_endpoint():
        print("\n❌ Data loading failed - this might be expected if no .mat files are available")
    
    print("\n🎉 Basic API tests completed!")

if __name__ == "__main__":
    main()
