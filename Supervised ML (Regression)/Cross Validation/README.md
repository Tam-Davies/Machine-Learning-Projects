# Boston Housing Regression with Cross-Validation

This project studies supervised regression on the Boston Housing dataset. The notebook compares a standardized linear regression baseline with Lasso and Ridge models, polynomial feature expansions, and cross-validated hyperparameter selection.

## Project Files

- [`notebook.ipynb`](notebook.ipynb): Complete analysis, model comparisons, regularization experiments, and grid search.
- `boston_housing_clean.pickle`: Cleaned Boston Housing data and its description. The notebook expects this file in the same directory.

## Workflow

1. Load the cleaned dataset from the pickle file.
2. Separate the predictors from `MEDV`, the median home-value target.
3. Create a shuffled three-fold cross-validation strategy.
4. Standardize the predictors and evaluate linear regression with cross-validated predictions.
5. Compare Lasso models over a geometric range of alpha values.
6. Add polynomial features and evaluate Lasso and Ridge regularization.
7. Inspect the coefficients selected by the best-performing pipeline.
8. Use `GridSearchCV` to select the polynomial degree and Ridge alpha jointly.

## Models and Evaluation

The notebook uses scikit-learn pipelines so scaling and feature generation are fitted within each cross-validation fold. Model quality is evaluated primarily with the coefficient of determination, $R^2$. The grid search explores:

- Polynomial degrees: 1, 2, and 3
- Ridge alpha values: 20 values geometrically spaced between 4 and 20
- Cross-validation: shuffled `KFold` with 3 splits and `random_state=72018`

The notebook also examines how Lasso regularization changes coefficient magnitude and sparsity, and how polynomial expansion affects model complexity.

## Requirements

- Python 3.x
- pandas
- NumPy
- Matplotlib
- scikit-learn

## Running the Notebook

Open `notebook.ipynb` in VS Code or Jupyter and run the cells from top to bottom. Start from the `Cross Validation` directory so the relative path to `boston_housing_clean.pickle` resolves correctly. The notebook is an exploratory analysis and does not export a fitted model or production prediction API.

## Notes

The Boston Housing dataset is used here for educational model-comparison purposes. Cross-validation scores can vary with the fold configuration, and the final in-sample score should not be interpreted as an unbiased estimate of generalization performance.
