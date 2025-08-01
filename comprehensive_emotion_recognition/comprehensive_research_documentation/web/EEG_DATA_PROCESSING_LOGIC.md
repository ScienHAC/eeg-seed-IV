# EEG Data Processing Logic - Identical to Model Training

## 🎯 **EXACT SAME LOGIC AS YOUR MODEL TRAINING**

This backend uses **IDENTICAL** processing logic as your `seed_iv_loader.py` model training:

### 📊 **Data Flow Overview**
```
SEED-IV .mat files (MATLAB)
    ↓
scipy.io.loadmat(file, struct_as_record=False, squeeze_me=True)
    ↓
Extract: de_LDS{trial} or de_movingAve{trial}
    ↓
3D Array: (62_channels, time_samples, 5_freq_bands)
    ↓
Reshape: data.transpose(1,0,2).reshape(time_samples, 62*5)
    ↓
Final: (time_samples, 310_features)
```

### 🔢 **Feature Structure Details**

#### **Both de_LDS and de_movingAve have SAME structure:**
- **310 features per time point**
- **62 EEG channels × 5 frequency bands = 310**

#### **5 Frequency Bands:**
1. **Delta (δ):** 1-4 Hz
2. **Theta (θ):** 4-8 Hz  
3. **Alpha (α):** 8-13 Hz
4. **Beta (β):** 13-30 Hz
5. **Gamma (γ):** 30-50 Hz

#### **62 EEG Channel Layout:**
Standard 10-20 system with additional channels for high-density recording

### 🧮 **Feature Extraction Logic**

#### **1. de_LDS (Differential Entropy - Linear Dynamic System):**
```
Source: de_LDS1.csv, de_LDS2.csv, ..., de_LDS24.csv
Structure: Each has 310 columns (62 channels × 5 bands)
Values: High precision like 27.795500626204074
```

#### **2. de_movingAve (Differential Entropy - Moving Average):**
```
Source: de_movingAve1.csv, de_movingAve2.csv, ..., de_movingAve24.csv  
Structure: Each has 310 columns (62 channels × 5 bands)
Values: High precision like 25.00743778857261
```

### 📈 **Frequency Band Processing**

#### **"All Bands" = SUM of all 5 frequency bands:**
```python
all_bands_sum = delta_avg + theta_avg + alpha_avg + beta_avg + gamma_avg
```

#### **Individual Bands = Average across 62 channels:**
```python
delta_avg = mean([Ch1_Delta, Ch2_Delta, ..., Ch62_Delta])
theta_avg = mean([Ch1_Theta, Ch2_Theta, ..., Ch62_Theta])
# ... and so on for each band
```

### 🔄 **3D Array Reshape (CRITICAL):**

Your model training uses this EXACT reshape logic:
```python
# Original: (62_channels, time_samples, 5_freq_bands)
data_3d = loadmat(file)['de_LDS1']  # Example

# Step 1: Transpose to (time_samples, channels, freq_bands)
transposed = data_3d.transpose(1, 0, 2)

# Step 2: Reshape to (time_samples, 310_features)
final_features = transposed.reshape(time_samples, 62 * 5)
```

This creates the **310-feature vectors** that your model trains on!

### 📋 **Data Verification**

#### **Backend matches your training:**
✅ Same .mat file loading with `scipy.io.loadmat`
✅ Same 3D → 2D reshape: `transpose(1,0,2).reshape(time, 310)`
✅ Same feature extraction: `de_LDS` and `de_movingAve`
✅ Same precision: High precision floating point values
✅ Same trial structure: 24 trials per session

#### **Feature Index Mapping:**
```
Feature 0-4:    Channel 1 (Delta, Theta, Alpha, Beta, Gamma)
Feature 5-9:    Channel 2 (Delta, Theta, Alpha, Beta, Gamma)
...
Feature 305-309: Channel 62 (Delta, Theta, Alpha, Beta, Gamma)
```

### 🎯 **Why This Matters**

Your model achieved **97.7% accuracy** using this exact data processing pipeline. The web interface now uses:

1. **Same .mat file sources**
2. **Same preprocessing steps**  
3. **Same 3D array handling**
4. **Same feature structure (310 features)**
5. **Same precision levels**

This ensures the web interface shows **real research data** processed identically to your training pipeline!
