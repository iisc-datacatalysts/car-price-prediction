# Predictive Modelling for Used Car Pricing

A comprehensive machine learning project for predicting used car prices using various regression models and advanced feature engineering techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Models Evaluated](#models-evaluated)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Key Insights](#key-insights)
- [Future Improvements](#future-improvements)

## 🎯 Overview

This project implements an end-to-end machine learning pipeline for predicting used car prices in the Indian market. The solution includes:

- **Comprehensive Data Cleaning**: Handling missing values, normalizing units, and extracting features from mixed-format columns
- **Enhanced Preprocessing**: 
  - **KNN Imputation**: Using K-Nearest Neighbors imputation (n_neighbors=5) for sophisticated missing value handling
  - **Polynomial Features**: Generating interaction features (degree=2, interaction_only=True) to capture feature relationships
  - Standard scaling for numerical features
- **Advanced Feature Engineering**: Creating derived features like car age, km/year, frequency encodings, and residual encodings
- **Multiple Model Evaluation**: Comparing baseline and advanced models including Ridge, Random Forest, XGBoost, CatBoost, and LightGBM
- **Target Encoding**: Using target encoding for high-cardinality categorical features
- **Quantile Regression**: Providing prediction intervals for uncertainty estimation
- **Model Interpretability**: SHAP analysis for feature importance
- **Production-Ready Inference**: Serialized models and inference function for deployment

## 📊 Dataset

The dataset contains information about **8,128 used cars** with the following attributes:

### Features

**Car Identification & Basic Info:**
- `name`: Full car name (combines brand and model)
- `year`: Manufacturing year
- `selling_price`: Target variable - the price at which the car was sold

**Car Usage History:**
- `km_driven`: Total kilometers driven
- `owner`: Ownership history (First Owner, Second Owner, etc.)
- `seller_type`: Type of seller (Individual, Dealer, Trustmark Dealer)

**Car Specifications:**
- `fuel`: Fuel type (Diesel, Petrol, LPG, CNG)
- `transmission`: Transmission type (Manual, Automatic)
- `mileage`: Fuel efficiency (with units: kmpl or km/kg)
- `engine`: Engine displacement (in CC)
- `max_power`: Maximum power output (in bhp)
- `torque`: Torque specification (in Nm or kgm, with RPM range)
- `seats`: Number of seats

**Dataset Source:** The dataset is loaded from the IISC Data Catalysts GitHub repository. The original dataset is maintained by Kaggle team - [Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho). This has been downloaded from Kaggle and uploaded into GitHub for ease of access.

## 🔧 Features

### Data Preprocessing
- **KNN Imputation**: Uses K-Nearest Neighbors (n_neighbors=5) to impute missing values based on similar observations (more sophisticated than median imputation)
- Missing value imputation (constant for categorical)
- Unit normalization (km/kg → kmpl, kgm → Nm)
- String parsing for mixed-format columns (mileage, engine, max_power, torque)
- RPM extraction from complex torque strings
- **Polynomial Features**: Generates interaction features (degree=2, interaction_only=True) to capture multiplicative relationships between features
- Standard scaling for numerical features

### Feature Engineering
- **Derived Features**: Car age, kilometers per year, abnormal usage flags
- **Frequency Encoding**: Normalized frequency of make and model occurrences
- **Residual Encoding**: Model-specific price deviations from overall mean
- **Log Transformation**: Applied to target variable to handle right-skewed distribution

### Model Training
- Stratified train-test split (85% train, 15% test)
- Cross-validation with honest target encoding
- Early stopping for gradient boosting models
- Hyperparameter tuning for optimal performance

## 📁 Project Structure

```
car-price-prediction/
├── DSPCourse_Project_UsedCarPricePrediction.ipynb  # Main notebook
├── README.md                                       # This file
├── CarData.csv                                     # Dataset (can be loaded from GitHub)
└── artifacts/                                      # Saved models (generated after running)
    ├── preprocessor_lgb.joblib                    # Preprocessing pipeline
    ├── target_encoder.joblib                       # Target encoder
    ├── lgb_model.txt                               # Main LightGBM model
    └── lgb_quantile_*.txt                          # Quantile regression models
```

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- Jupyter Notebook or JupyterLab

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd DSP
   ```

2. **Install required packages:**
   
   The notebook will automatically install packages in the first cell, or you can install them manually:
   
   ```bash
   pip install shap lightgbm category_encoders catboost scikit-learn pandas numpy matplotlib seaborn
   ```

3. **Open the notebook:**
   ```bash
   jupyter notebook DSPCourse_Project_UsedCarPricePrediction.ipynb
   ```

## 💻 Usage

### Running the Notebook

1. Open the notebook in Jupyter
2. Run all cells sequentially (Cell → Run All)
3. The notebook will:
   - Load data from GitHub
   - Perform data cleaning and feature engineering
   - Train multiple models
   - Evaluate performance
   - Save model artifacts to `artifacts/` directory

### Making Predictions

After training, you can use the `predict_single()` function to predict prices for new cars:

```python
# Example usage
example_car = {
    'name': 'Maruti Swift Dzire VDI',
    'year': 2014,
    'kms_driven': 145500,
    'fuel': 'Diesel',
    'transmission': 'Manual',
    'owner': 'First Owner',
    'seller_type': 'Individual',
    'mileage': '23.4 kmpl',
    'engine': '1248 CC',
    'max_power': '74 bhp',
    'torque': '190Nm@ 2000rpm',
    'seats': 5.0
}

predicted_price = predict_single(example_car)
print(f"Predicted Price: ₹{predicted_price:,.0f}")
```

## 📈 Methodology

### Data Cleaning Pipeline

1. **Name Parsing**: Extract make (brand) and model from car names
2. **Mileage Normalization**: Convert km/kg to kmpl using fuel density ratios
3. **Torque Processing**: Extract torque values and normalize units (kgm → Nm)
4. **RPM Extraction**: Parse RPM information from various formats (ranges, +/- notation)

### Enhanced Preprocessing Pipeline

1. **KNN Imputation**: Uses K-Nearest Neighbors (n_neighbors=5) to impute missing values. This is more sophisticated than median imputation as it considers relationships between features to provide more accurate missing value estimates.
2. **Standard Scaling**: Normalizes features to have zero mean and unit variance
3. **Polynomial Features**: Creates interaction features (degree=2, interaction_only=True) to capture multiplicative relationships between features (e.g., age × kms_driven, age × max_power_value). The `interaction_only=True` parameter means only interaction terms are created, not squared terms.

### Feature Engineering

1. **Temporal Features**: Calculate car age from manufacturing year
2. **Usage Features**: Compute kilometers per year and flag abnormal usage
3. **Encoding Features**: 
   - Frequency encoding for make/model popularity
   - Residual encoding for model-specific price patterns
4. **Target Transformation**: Apply log1p transformation to handle skewness

### Model Training Strategy

1. **Baseline Models**: Start with simple linear models (Ridge) with enhanced preprocessing (KNN imputation + polynomial features)
2. **Tree-Based Models**: Evaluate Random Forest, XGBoost, CatBoost, LightGBM
3. **Ensemble Methods**: Stack multiple models for improved performance
4. **Advanced Techniques**: 
   - Target encoding for high-cardinality features (make, model)
   - Quantile regression for uncertainty estimation
   - Cross-validation with honest target encoding (out-of-fold predictions)
   - Early stopping for gradient boosting models

## 🤖 Models Evaluated

| Model | Description | Key Features |
|-------|-------------|--------------|
| **Ridge Regression** | Linear baseline model | L2 regularization, fast training |
| **Random Forest** | Ensemble of decision trees | Non-linear relationships, feature interactions |
| **XGBoost** | Gradient boosting framework | Regularization, subsampling |
| **CatBoost** | Categorical boosting | Native categorical handling |
| **LightGBM** | Gradient boosting framework | Fast training, low memory usage |
| **Stacking Ensemble** | Meta-learner combining models | Combines predictions from multiple models |
| **LightGBM + Target Encoding** | Advanced model with encoding | Best performance, handles high-cardinality features |

## 📊 Results

### Model Performance

The LightGBM model with target encoding achieved the best performance:

- **Training R²**: 0.984
- **Test R²**: 0.950
- **RMSE**: ₹102,484
- **MAE**: ₹57,977

### Cross-Validation Results

5-fold cross-validation R² scores:
- **Ridge Regression**: 0.84 - 0.94 (varies by fold)
- **CatBoost**: 0.94 - 0.95
- **LightGBM**: 0.94 - 0.95
- **XGBoost**: 0.93 - 0.95

### Key Findings

1. **Strong Predictors**:
   - Car age (negative correlation: -0.71)
   - Kilometers driven (negative correlation: -0.25)
   - Max power and engine size (positive correlations)

2. **Categorical Insights**:
   - Diesel cars command higher prices than Petrol/LPG/CNG
   - Automatic transmission increases price
   - First owner cars are most valuable

3. **Prediction Intervals**:
   - 90% prediction interval coverage: ~77.0%
   - Quantile regression provides uncertainty estimates (5th, 50th, 95th percentiles)

## 🛠 Technologies Used

- **Python 3.10+**
- **Data Science Libraries**:
  - `pandas` - Data manipulation
  - `numpy` - Numerical computing
  - `scikit-learn` - Machine learning
- **Advanced ML Libraries**:
  - `lightgbm` - Gradient boosting
  - `xgboost` - Gradient boosting
  - `catboost` - Categorical boosting
  - `category_encoders` - Target encoding
- **Visualization**:
  - `matplotlib` - Plotting
  - `seaborn` - Statistical visualization
  - `shap` - Model interpretability
- **Utilities**:
  - `joblib` - Model serialization

## 🔍 Key Insights

1. **Data Quality**: The dataset contains missing values in performance attributes (~221 missing in mileage, engine, max_power, torque, seats), which are handled through KNN imputation (n_neighbors=5) - a more sophisticated approach than median imputation

2. **Feature Importance**: Car age is the strongest predictor, followed by kilometers driven and engine specifications

3. **Model Selection**: LightGBM with target encoding outperforms other models, balancing accuracy and training time

4. **Target Transformation**: Log transformation is crucial for handling the right-skewed price distribution

5. **High-Cardinality Features**: Target encoding effectively handles the 2,058 unique car names and models

## 🔮 Future Improvements

- **Enhanced Name Parsing**: Use regex patterns or fuzzy matching against curated make/model databases for more accurate extraction
- **Leakage Testing**: Evaluate models with and without potentially leaked features to ensure robustness
- **Calibrated Intervals**: Implement conformalized quantile regression for better-calibrated uncertainty estimates
- **Additional Features**: Explore brand reputation scores, depreciation curves, market segment indicators
- **Polynomial Feature Selection**: Consider feature selection techniques to reduce dimensionality after polynomial feature generation
- **KNN Imputation Tuning**: Experiment with different values of n_neighbors for KNN imputation to optimize missing value handling
- **Model Optimization**: Hyperparameter tuning with Optuna or similar tools
- **Testing**: Add unit tests for preprocessing pipeline and inference function before production deployment
- **Computational Efficiency**: Monitor training time with polynomial features; consider feature selection if dimensionality becomes too high
- **Deployment**: Create API endpoint for real-time predictions

## 📝 Notes

- The dataset is automatically loaded from the IISC Data Catalysts GitHub repository
- Python 3.10+ is recommended for running this notebook
- Required packages are installed in the first code cell of the notebook
- All models are saved in the `artifacts/` directory after training
- The notebook includes comprehensive documentation and comments
- The enhanced preprocessing pipeline uses KNN imputation and polynomial interaction features
- Results may vary slightly due to random seeds and data splits

## 👤 Authors

- **Vimalraj Kanagaraj** (vimalrajk@iisc.ac.in)
- **Anfaal Obaid Waafy** (anfaalwaafy@iisc.ac.in)
- **Manikanda Sakthi Subramaniam**(manikandasa1@iisc.ac.in)
- **Abhilasha Kawle**(abhilashak@iisc.ac.in)

## Dataset: [Kaggale](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)

## 📄 License

This project is part of a course assignment. Please refer to the course guidelines for usage terms.

## 🙏 Acknowledgments

- Dataset provided via GitHub repository
- Course instructors and materials
- Open-source ML community for excellent libraries

---

**Note**: This project is for educational purposes. Model performance may vary with different datasets or market conditions.

