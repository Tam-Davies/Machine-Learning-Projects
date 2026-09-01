# Car Price Prediction

## Project Overview
This project implements a **machine learning regression model** to predict car prices based on various car features and characteristics. Using supervised learning with linear regression, the model analyzes 24 numeric features from car data to accurately estimate vehicle prices.

## Objective
Build and evaluate a predictive model that can estimate car prices with high accuracy using linear regression and feature engineering techniques.

## Dataset
- **Source**: CarPrice.csv
- **Total Records**: Multiple car samples with comprehensive features
- **Features**: 25 features (24 predictors + 1 target)
- **Target Variable**: `price` (car price in dollars)

## Key Features Included
- Numeric features only (non-numeric features excluded)
- Features include: engine specifications, dimensions, weight, and performance metrics
- Car ID and Symboling removed as they are identification-only features

## Project Structure
```
Car Price Prediction/
├── README.md                          # Project documentation (this file)
├── data/
│   └── CarPrice.csv                   # Original dataset
├── notebooks/
│   └── main.ipynb                     # Main analysis and modeling notebook
```

## Methodology

### 1. **Data Preprocessing**
   - Load and inspect the CarPrice dataset
   - Filter for numeric features only
   - Remove identification columns (Car_ID, Symboling)
   - Handle missing values and outliers

### 2. **Exploratory Data Analysis (EDA)**
   - Statistical summary of all features
   - Correlation analysis with target variable
   - Distribution analysis using:
     - Histograms
     - Box plots
     - Scatter plots
   - Box-Cox transformation for normality checking
   - Normality tests for continuous variables

### 3. **Feature Engineering**
   - Correlation analysis to identify key predictors
   - Selection of features with strong relationship to price
   - Log transformation of skewed distributions
   - Feature normalization and scaling

### 4. **Model Development**
   - **Algorithm**: Linear Regression
   - **Train-Test Split**: 80-20 or custom split
   - **Data Scaling**: StandardScaler for feature normalization
   - **Pipeline Implementation**: Automated preprocessing → modeling workflow

### 5. **Model Evaluation**
   - **Metrics Used**:
     - Mean Squared Error (MSE)
     - Mean Absolute Error (MAE)
     - Root Mean Squared Error (RMSE)
     - R² Score (Coefficient of Determination)
   - **Residual Analysis**:
     - Residual plots to check for patterns
     - Q-Q plots for normality assessment
     - Homoscedasticity validation

## Key Results
- Model predictions evaluated on test set
- Performance metrics calculated to assess model accuracy
- Residual diagnostics performed to validate model assumptions
- Feature coefficients extracted to understand price drivers

## Technologies Used
- **Python 3.x**
- **Libraries**:
  - pandas: Data manipulation and analysis
  - numpy: Numerical computations
  - scikit-learn: Machine learning (Linear Regression, Pipeline, Scaling, Metrics)
  - matplotlib: Data visualization
  - seaborn: Statistical visualizations
  - scipy: Statistical tests and transformations

## Model Pipeline
```
Input Data → Scaling (StandardScaler) → Linear Regression → Predictions
```

## Files Description

### main.ipynb
Complete Jupyter notebook containing:
1. Module imports and dependencies
2. Data loading and inspection
3. Exploratory data analysis
4. Feature selection and engineering
5. Train-test data split
6. Model training and evaluation
7. Residual analysis
8. Pipeline implementation
9. Final predictions and metrics

## Usage

### Running the Notebook
1. Navigate to the `notebooks/` directory
2. Open `main.ipynb` in Jupyter Notebook or JupyterLab
3. Execute cells sequentially to reproduce the analysis
4. Modify the data path if needed to match your local environment

### Making Predictions
```python
# Load the trained pipeline
# car_price_predictions = pipe.predict(new_data)
```

## Performance Insights
- The model achieves quantifiable R² score on test data
- RMSE indicates average prediction error magnitude
- MAE provides mean absolute prediction error
- Residual analysis validates linear regression assumptions

## Future Improvements
- Explore polynomial and interaction features
- Test other regression algorithms (Ridge, Lasso, ElasticNet)
- Implement cross-validation for robust evaluation
- Hyperparameter tuning
- Feature importance analysis using coefficients
- Consider ensemble methods for better predictions

## Notes
- All paths in the notebook are absolute; adjust if running on different machine
- Ensure all required libraries are installed: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`
- The model assumes linear relationship between features and price

## Author
Data Scientist / ML Engineer

## Date
2024-2025

---

**Last Updated**: September 2025
