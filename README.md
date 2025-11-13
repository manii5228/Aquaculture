Aquaculture 
============

This repository contains machine learning models for predicting various parameter levels in aquaculture environments using various deep learning and statistical approaches.


Models Implemented
------------------

### 1. **Convolutional Neural Network (CNN)**

*   **Purpose**: Spatial pattern recognition in salinity data
    
*   **Architecture**:
    
    *   Multiple Conv2D layers with BatchNormalization
        
    *   MaxPooling for dimensionality reduction
        
    *   Fully connected layers for regression
        
### 2. **CNN-LSTM Hybrid Model**

*   **Purpose**: Spatiotemporal salinity prediction
    
*   **Architecture**:
    
    *   TimeDistributed CNN for spatial feature extraction
        
    *   LSTM layers for temporal sequence modeling
        
    *   Sequence-to-one prediction
        
### 3. **U-Net Architecture**

*   **Purpose**: Image-to-image salinity prediction
    
*   **Architecture**:
    
    *   Encoder-decoder structure with skip connections
        
    *   Detailed spatial feature preservation
        
    *   Pixel-wise salinity prediction

### 4. **Long Short-Term Memory (LSTM)**

*   **Purpose**: Temporal sequence prediction
    
*   **Architecture**:
    
    *   Multiple LSTM layers with dropout
        
    *   Sequence-to-value prediction
        

### 5. **Gated Recurrent Unit (GRU)**

*   **Purpose**: Efficient temporal prediction
    
*   **Architecture**:
    
    *   GRU layers with simplified gating mechanism
        
    *   Lower computational complexity than LSTM
          

### 6. **Artificial Neural Network (ANN)**

*   **Purpose**: Baseline regression model
    
*   **Architecture**:
    
    *   Fully connected layers with regularization
        
    *   Simple feedforward network

### 7. **Generalized Additive Model (GAM)**

*   **Purpose**: Interpretable statistical modeling
    
*   **Architecture**:
    
    *   Smoothing splines for non-linear relationships
        
    *   Additive component modeling
        
    *   Feature importance analysis
        

Dataset
-------

The models use data stored in NetCDF format with the following characteristics:

*   **Variable**: SALT (salinity), pH, Dissolved Oxygen, Turbidity
    
*   **Dimensions**: Time × Depth × Latitude × Longitude
    
*   **Preprocessing**:
    
    *   Downsampling (4x reduction)
        
    *   Orientation correction
        
    *   Missing value imputation (median strategy)
        
    *   Global standardization
        

Installation
------------

### Prerequisites

  # Clone the repository  
  ```
  git clone https://github.com/yourusername/aquaculture.git
  cd aquaculture
# Install required packages
  pip install -r requirements.txt
 ```

### Required Packages

`   xarray  numpy  scikit-learn  tensorflow  matplotlib  seaborn  pygam  netcdf4   `



### Model Configuration

Key parameters for each model:

*   **Sequence Length**: 3-30 time steps
    
*   **Train/Val/Test Split**: 60%/20%/20%
    
*   **Batch Size**: 32-64
    
*   **Epochs**: 50-200 (with early stopping)
    
*   **Evaluation Metrics**: MSE, RMSE, MAE, R²
    

Performance Metrics
-------------------

All models are evaluated using:

### Regression Metrics

*   **MSE** (Mean Squared Error)
    
*   **RMSE** (Root Mean Squared Error)
    
*   **MAE** (Mean Absolute Error)
    
*   **R²** (R-squared)
    

### Classification Metrics (Derived)

*   **Accuracy**, **Precision**, **Recall** (via thresholding)
