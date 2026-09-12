# Regularization for Boston Housing Regression

This project explores feature scaling, polynomial expansion, and regularization for predicting Boston housing values. The notebook compares ordinary Linear Regression with Ridge and Lasso models and examines how regularization changes coefficient magnitude, sparsity, and predictive performance.

## Project Files

- [`notebook.ipynb`](notebook.ipynb): End-to-end analysis of scaling, polynomial features, Ridge, Lasso, and held-out evaluation.
- `boston_housing_clean.pickle`: Cleaned Boston Housing data and its description. The notebook expects this file in the same directory.

## Workflow

1. Load the cleaned dataset from the pickle file.
2. Separate `MEDV` as the target and use the remaining columns as predictors.
3. Standardize features with `StandardScaler` and verify the transformation manually with NumPy.
4. Fit ordinary Linear Regression and inspect coefficients before and after scaling.
5. Generate second-degree polynomial features and fit Lasso models with multiple alpha values.
6. Split the polynomial features into training and test sets using `random_state=72018`.
7. Compare held-out $R^2$ scores and coefficient sparsity for Linear Regression, Lasso, and Ridge.
8. Evaluate a final baseline using standardized original features without polynomial expansion.

## Models and Evaluation

The notebook uses:

- `LinearRegression` as the unregularized baseline.
- `Ridge(alpha=0.001)` to shrink coefficients without forcing them to zero.
- `Lasso` with default settings, plus `alpha=0.1`, `alpha=1`, and `alpha=0.001` experiments.
- Second-degree polynomial features with `include_bias=False`.

Model performance is measured with the coefficient of determination, $R^2$. Coefficient magnitude is summarized with the sum of absolute coefficients, while sparsity is assessed by counting coefficients that are not equal to zero.

## Requirements

- Python 3.x
- pandas
- NumPy
- Matplotlib
- scikit-learn

## Running the Notebook

Open `notebook.ipynb` in VS Code or Jupyter and run the cells from top to bottom. Start from the `Regularization` directory so the relative path to `boston_housing_clean.pickle` resolves correctly.

The notebook is an educational model-comparison study. It reports exploratory and held-out scores but does not save a fitted model or provide a production inference pipeline.

## Notes

Scaling is important for regularized models because the penalty acts on coefficient values. The train/test section fits the scaler on the training data and applies the same transformation to the test data. Earlier exploratory cells also fit transformations on the full dataset to illustrate the mechanics of scaling and coefficient comparison.
