# Regularization and Gradient Descent for Regression

This project explores how polynomial complexity, coefficient regularization, feature scaling, and gradient-descent settings influence regression models. The notebook contains a visual sinusoid experiment followed by an Ames housing price comparison, with markdown sections explaining the purpose of each modeling stage.

## Project Files

- [`notebook.ipynb`](notebook.ipynb): Complete notebook with visual experiments, preprocessing, model fitting, and RMSE comparisons.
- `X_Y_Sinusoid_Data.csv`: Sparse noisy observations used to demonstrate polynomial regression.
- `Ames_Housing_Sales.csv`: Ames housing data used for price prediction.

## Notebook Sections

The notebook is organized from visual intuition to tabular evaluation: it first demonstrates coefficient instability on a high-degree polynomial, then applies the same regularization ideas to housing-price prediction and compares batch-style estimators with stochastic gradient descent.

### 1. Polynomial Regression on a Sinusoid

A degree-20 polynomial is fitted to noisy sinusoid observations. The learned curve is compared with the underlying function to illustrate the effect of high model complexity.

### 2. Ridge and Lasso Regularization

Ridge and Lasso regression are fitted to the polynomial features. The notebook compares predictions and coefficient magnitudes, showing how penalties can reduce instability and encourage simpler models.

### 3. Ames Housing Preprocessing

The Ames data is one-hot encoded, split into training and test sets with `random_state=42`, and checked for skewed floating-point features. Features exceeding an absolute skew threshold of `0.75` are transformed with `log1p` using the training data.

### 4. Cross-Validated Model Comparison

The target is `SalePrice`. A held-out test set is evaluated with root mean squared error (RMSE). The notebook compares:

- Linear Regression
- `RidgeCV` with four-fold internal cross-validation
- `LassoCV` with three-fold internal cross-validation
- `ElasticNetCV` across candidate alpha values and L1 ratios

The results are collected in an RMSE table and visualized as actual-versus-predicted prices.

### 5. Stochastic Gradient Descent

`SGDRegressor` is used with the parameter settings selected by the regularized models. The notebook compares default optimization, a fixed learning rate, and MinMax-scaled features to examine how feature scale and learning-rate choices affect RMSE.

## Requirements

- Python 3.x
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## Running the Notebook

Open `notebook.ipynb` in VS Code or Jupyter and run the cells from top to bottom. Start from the `Regularization and Gradient Descent` directory so both CSV paths resolve correctly. The notebook is an educational model-comparison study and does not export a fitted model or production inference pipeline.

## Evaluation Notes

RMSE is reported on the Ames test split, so lower values indicate smaller prediction errors. Results may vary with library versions and optimizer convergence. The sinusoid section is intended for visualization, while the Ames section provides the tabular regression comparison.
