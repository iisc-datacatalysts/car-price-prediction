# Used Car Price Prediction

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

**Dataset Source:** [GitHub Repository](https://raw.githubusercontent.com/vimal-kanagaraj/dataset/main/CarData.csv)

## 🔧 Features

### Data Preprocessing
- Missing value imputation (median for numeric, constant for categorical)
- Unit normalization (km/kg → kmpl, kgm → Nm)
- String parsing for mixed-format columns (mileage, engine, max_power, torque)
- RPM extraction from complex torque strings

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
DSP/
├── DSPCourse_Project_UsedCarPricePrediction-4_Updated.ipynb  # Main notebook
├── README.md                                                  # This file
└── artifacts/                                                 # Saved models (generated after running)
    ├── preprocessor_lgb.joblib                               # Preprocessing pipeline
    ├── target_encoder.joblib                                  # Target encoder
    ├── lgb_model.txt                                          # Main LightGBM model
    └── lgb_quantile_*.txt                                     # Quantile regression models
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
   jupyter notebook DSPCourse_Project_UsedCarPricePrediction-4_Updated.ipynb
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

### Feature Engineering

1. **Temporal Features**: Calculate car age from manufacturing year
2. **Usage Features**: Compute kilometers per year and flag abnormal usage
3. **Encoding Features**: 
   - Frequency encoding for make/model popularity
   - Residual encoding for model-specific price patterns
4. **Target Transformation**: Apply log1p transformation to handle skewness

### Model Training Strategy

1. **Baseline Models**: Start with simple linear models (Ridge)
2. **Tree-Based Models**: Evaluate Random Forest, XGBoost, CatBoost, LightGBM
3. **Ensemble Methods**: Stack multiple models for improved performance
4. **Advanced Techniques**: 
   - Target encoding for high-cardinality features
   - Quantile regression for uncertainty estimation
   - Cross-validation with honest encoding

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

- **Training R²**: 0.974
- **Test R²**: 0.949
- **RMSE**: ₹104,401
- **MAE**: ₹58,607

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
   - 90% prediction interval coverage: ~76.6%
   - Quantile regression provides uncertainty estimates

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

1. **Data Quality**: The dataset contains missing values in performance attributes (~221 missing), which are handled through imputation

2. **Feature Importance**: Car age is the strongest predictor, followed by kilometers driven and engine specifications

3. **Model Selection**: LightGBM with target encoding outperforms other models, balancing accuracy and training time

4. **Target Transformation**: Log transformation is crucial for handling the right-skewed price distribution

5. **High-Cardinality Features**: Target encoding effectively handles the 2,058 unique car names and models

## 🔮 Future Improvements

- **Enhanced Name Parsing**: Use regex patterns or fuzzy matching against curated make/model databases
- **Leakage Testing**: Evaluate models with and without potentially leaked features
- **Calibrated Intervals**: Implement conformalized quantile regression for better uncertainty estimates
- **Additional Features**: Explore brand reputation scores, depreciation curves, market segment indicators
- **Model Optimization**: Hyperparameter tuning with Optuna or similar tools
- **Testing**: Add unit tests for preprocessing pipeline and inference function
- **Deployment**: Create API endpoint for real-time predictions

## 📝 Notes

- The dataset is automatically loaded from a GitHub repository
- All models are saved in the `artifacts/` directory after training
- The notebook includes comprehensive documentation and comments
- Results may vary slightly due to random seeds and data splits

## 👤 Authors

- **Vimalraj Kanagaraj** (mailto:vimalrajk@iisc.ac.in)
- **Anfaal Obaid Waafy** (mailto:anfaalwaafy@iisc.ac.in)
- **Manikanda Sakthi Subramaniam**(mailto:manikandasa1@iisc.ac.in)
- **Abhilasha Kawle**(mailto:abhilashak@iisc.ac.in)

- Dataset: [Kaggale](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)

## 📄 License

This project is part of a course assignment. Please refer to the course guidelines for usage terms.

## 🙏 Acknowledgments

- Dataset provided via GitHub repository
- Course instructors and materials
- Open-source ML community for excellent libraries

---

**Note**: This project is for educational purposes. Model performance may vary with different datasets or market conditions.

