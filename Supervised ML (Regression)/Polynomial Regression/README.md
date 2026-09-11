# Polynomial Regression for Car Price Prediction

## Project Overview
This project applies polynomial regression to predict car prices using a preprocessed vehicle dataset. The notebook explores how a simple linear model performs compared to polynomial transformations, model scaling, and hyperparameter tuning.

The goal is to understand whether nonlinear relationships between vehicle characteristics and price can be captured more accurately by adding polynomial terms.

## Business Problem
Car price depends on several interconnected factors, including:
- engine size
- horsepower
- curb weight
- car length
- wheelbase
- categorical design attributes such as transmission and body type

Because these relationships are often nonlinear, a standard linear regression model may underfit the data. Polynomial regression allows the model to capture curvature and interaction effects in the feature space.

## Dataset
The project uses the encoded car dataset stored in:
- `data/encoded_car_data.csv`

The dataset contains numerical features that were already transformed for model use. The target variable is:
- `price`

## Workflow
### 1. Data Loading and Initial Inspection
The notebook begins by loading the dataset and checking the structure using `data.info()`. This step confirms column types, missing values, and the overall shape of the dataset.

### 2. Exploratory Visualization
A few key relationships are plotted using Seaborn:
- `curbweight` vs `price`
- `carlength` vs `price`
- `horsepower` vs `price`

These plots help identify whether a linear pattern exists or whether a nonlinear trend is more appropriate.

### 3. Train-Test Split
The data is split into training and testing sets using `train_test_split` with a fixed random state for reproducibility.

This step ensures the model is evaluated on unseen data and helps prevent overfitting from being hidden by training performance alone.

### 4. Baseline Linear Regression
A simple `LinearRegression` model is trained on the training set and evaluated using R²:
- training R²
- testing R²

The notebook also compares actual versus fitted values using a distribution plot to visually assess model fit.

### 5. Coefficient Analysis
The model coefficients are inspected to understand which features contribute the most to price prediction. A bar chart of absolute coefficients helps highlight the dominant variables.

### 6. Feature-by-Feature R² Comparison
The function `get_R2_features()` evaluates each feature individually by fitting the same model on a single variable and recording the train/test R² values.

This helps assess how strongly each feature explains the target on its own.

### 7. Scaling with a Pipeline
A `Pipeline` is created with:
- `StandardScaler()`
- `LinearRegression()`

This step checks whether standardizing features improves model performance or stabilizes the learning process.

### 8. Polynomial Feature Transformation
The notebook creates polynomial features using:
- `PolynomialFeatures(degree=2, include_bias=False)`

The transformed training and testing arrays are then used to fit a linear model on the expanded feature space. This allows the model to capture nonlinear relationships between predictors and price.

### 9. Pipeline with Polynomial Features
A second pipeline combines:
- polynomial transformation
- linear regression

This form is cleaner and easier to reuse, especially when testing multiple configurations.

### 10. Hyperparameter Tuning with GridSearchCV
The final stage uses `GridSearchCV` to tune the polynomial degree and, where applicable, model settings.

This step searches across candidate configurations and selects the best-performing estimator based on validation behavior.

## Key Metrics
The notebook evaluates model quality using the coefficient of determination, R²:

- R² close to 1 indicates a strong fit
- lower R² on test data may suggest overfitting or insufficient model complexity
- comparing train vs test performance provides a practical model-selection signal

## Tools and Libraries
This project uses:
- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- tqdm

## Project Structure
```text
Polynomial Regression/
├── data/
│   └── encoded_car_data.csv
├── notebook.ipynb
├── README.md
└── ...
```

## How to Run
1. Open the notebook in Jupyter or VS Code.
2. Ensure all required libraries are installed.
3. Run the cells in sequence.
4. Review the generated plots and printed R² values.

Typical dependencies include:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tqdm
```

## Notes
This project is a practical example of how polynomial regression can improve prediction when linear assumptions are too restrictive. It demonstrates the full workflow from data preparation to model validation and tuning.

## Conclusion
The notebook compares multiple modeling strategies and shows that polynomial transformations can improve predictive performance when the data contains nonlinear relationships. The project is useful for learning regression modeling, feature engineering, and model evaluation concepts in a real-world dataset.
