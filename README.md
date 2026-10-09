## Model Architecture & Training

### Input Data
- Features: 43-element numpy arrays (model level precipitation classification)
  - Elements 0-19: Surface + 19 model level temperatures (T)
  - Elements 20-39: Surface + 19 model level relative humidity (%)
  - Elements 40-42: Normalized Latitude, Longitude, Elevation (optional)
- Labels: 4-class (0=Rain, 1=Snow, 2=Freezing Rain, 3=Ice Pellets)

### LightGBM Parameters
- num_leaves: 63
- learning_rate: 0.03
- max_depth: -1
- min_child_samples: 80
- max_bins: 300
- importance_type: 'gain'
- Default class weights: {0: 1, 1: 1, 2: 2, 3: 4}

### Cross-Validation
- Method: 10-fold stratified K-fold
- Feature importance: Average gain across all folds

## Requirements

### Dependencies
- numpy>=1.20.0
- scikit-learn>=1.0.0
- lightgbm>=3.3.0
- joblib>=1.0.0
- pandas>=1.3.0
- scipy>=1.7.0
