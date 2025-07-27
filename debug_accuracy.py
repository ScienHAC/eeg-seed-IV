"""
EMERGENCY DEBUG: Find why accuracy is terrible
==============================================

Let's test the simplest possible approach to isolate the problem.
If we can't beat random chance (25%), something is fundamentally wrong.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def load_minimal_seed_data():
    """Load data in the simplest possible way"""
    print("🔍 MINIMAL DATA LOADING TEST")
    print("=" * 50)
    
    all_X = []
    all_y = []
    
    # Test with just 1 session, 3 subjects to start
    for session in [1]:
        for subject in [1, 2, 3]:
            print(f"📂 Session {session}, Subject {subject}")
            
            # Try de_LDS first (it was slightly better)
            for trial in range(1, 25):  # 24 trials per subject
                file_path = Path(f"csv/{session}/{subject}/de_LDS{trial}.csv")
                
                if file_path.exists():
                    try:
                        # CRITICAL FIX: Skip header row!
                        data = pd.read_csv(file_path, header=0).values  # header=0 means first row is header
                        
                        # Get emotion label from trial number (SEED-IV mapping)
                        # Trials 1-6: Neutral (0), 7-12: Sad (1), 13-18: Fear (2), 19-24: Happy (3)
                        if 1 <= trial <= 6:
                            emotion = 0  # Neutral
                        elif 7 <= trial <= 12:
                            emotion = 1  # Sad  
                        elif 13 <= trial <= 18:
                            emotion = 2  # Fear
                        elif 19 <= trial <= 24:
                            emotion = 3  # Happy
                        else:
                            continue
                        
                        # Average across time dimension (if exists)
                        if len(data.shape) > 1:
                            feature_vector = np.mean(data, axis=0)  # Average across time (rows)
                        else:
                            feature_vector = data
                            
                        all_X.append(feature_vector)
                        all_y.append(emotion)
                        
                    except Exception as e:
                        print(f"   ❌ Trial {trial}: {e}")
                        continue
    
    X = np.array(all_X)
    y = np.array(all_y)
    
    print(f"\n✅ Loaded: {len(X)} samples, {X.shape[1]} features")
    print(f"📊 Label distribution: {Counter(y)}")
    print(f"🎯 Expected random accuracy: 25%")
    
    return X, y

def test_simple_classifiers(X, y):
    """Test with the absolute simplest setup"""
    print(f"\n🧪 SIMPLE CLASSIFIER TESTS")
    print("=" * 50)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test classifiers
    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42)
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        print(f"\n🔬 Testing {name}...")
        
        # Train
        clf.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = clf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = accuracy
        
        print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        if accuracy < 0.30:
            print("   🔥 WORSE THAN RANDOM! Something is very wrong!")
        elif accuracy < 0.40:
            print("   🟠 Still worse than expected")
        elif accuracy > 0.50:
            print("   🟢 Better than random - getting somewhere!")
        
        # Show detailed report for best result
        if accuracy == max(results.values()):
            print(f"\n📋 Best result details ({name}):")
            print(classification_report(y_test, y_pred, 
                                      target_names=['Neutral', 'Sad', 'Fear', 'Happy']))
    
    return results

def test_label_mapping():
    """Test if our label mapping is correct"""
    print(f"\n🏷️ LABEL MAPPING VERIFICATION")
    print("=" * 50)
    
    # Check a few files manually
    test_cases = [
        (1, 1, 3, 0),   # Should be Neutral
        (1, 1, 9, 1),   # Should be Sad
        (1, 1, 15, 2),  # Should be Fear
        (1, 1, 21, 3),  # Should be Happy
    ]
    
    for session, subject, trial, expected_label in test_cases:
        file_path = Path(f"csv/{session}/{subject}/de_LDS{trial}.csv")
        
        if file_path.exists():
            data = pd.read_csv(file_path, header=0).values  # Skip header
            print(f"📄 Session {session}, Subject {subject}, Trial {trial}")
            print(f"   Expected emotion: {['Neutral', 'Sad', 'Fear', 'Happy'][expected_label]}")
            print(f"   Data shape: {data.shape}")
            
            # Convert to numeric if needed
            if data.dtype == 'object':
                print(f"   ⚠️ Data contains non-numeric values!")
            else:
                print(f"   Data range: {data.min():.3f} to {data.max():.3f}")
                print(f"   Data mean: {data.mean():.3f}")
        else:
            print(f"❌ File not found: {file_path}")

def main():
    """Main debugging function"""
    print("🚨 EMERGENCY ACCURACY DEBUG")
    print("=" * 60)
    print("🎯 Goal: Find why we can't beat 25% random chance")
    print("🔍 Testing minimal setup to isolate the problem")
    
    # Step 1: Load minimal data
    X, y = load_minimal_seed_data()
    
    if len(X) == 0:
        print("❌ No data loaded - check file paths")
        return
    
    # Step 2: Verify label mapping
    test_label_mapping()
    
    # Step 3: Test simple classifiers
    results = test_simple_classifiers(X, y)
    
    # Step 4: Analysis
    print(f"\n" + "=" * 60)
    print("🔍 DEBUGGING CONCLUSIONS")
    print("=" * 60)
    
    best_accuracy = max(results.values())
    best_method = max(results.keys(), key=lambda k: results[k])
    
    print(f"🏆 Best result: {best_accuracy:.3f} ({best_method})")
    
    if best_accuracy < 0.30:
        print("🔥 CRITICAL: Much worse than random (25%)")
        print("   Possible causes:")
        print("   - Wrong label mapping")
        print("   - Data preprocessing error")
        print("   - Feature extraction bug")
        print("   - File reading issue")
    elif best_accuracy < 0.40:
        print("🟠 CONCERNING: Still below expected")
        print("   - Might be overfitting")
        print("   - Need better features")
    elif best_accuracy > 0.50:
        print("🟢 PROMISING: Better than random!")
        print("   - Basic approach works")
        print("   - Need to scale up carefully")
    
    print(f"\n🎯 NEXT STEPS:")
    if best_accuracy < 0.35:
        print("   1. Check label mapping (most likely issue)")
        print("   2. Verify data file structure")
        print("   3. Test with different feature extraction")
    else:
        print("   1. Scale up to more subjects")
        print("   2. Improve feature selection")
        print("   3. Try ensemble methods")

if __name__ == "__main__":
    main()
