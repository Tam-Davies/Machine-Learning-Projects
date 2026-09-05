# Ames House Pricing

This project develops linear regression models to predict residential sale prices in Ames, Iowa. It compares numeric features with and without one-hot encoding and examines how feature scaling affects test-set error.

## Project Files

- [`notebook.ipynb`](notebook.ipynb): Exploratory analysis, preprocessing, model comparison, and prediction visualization.
- `Ames_Housing_Sales.csv`: Housing records used for the analysis.

## Workflow

1. Load the Ames housing sales data and inspect the initial records.
2. Identify categorical variables with more than one observed level.
3. Compare the original numeric data with a one-hot encoded representation.
4. Split both representations into aligned training and test sets.
5. Fit linear regression models and compare mean squared error.
6. Evaluate standard, min-max, and max-absolute scaling on numeric features.
7. Visualize predicted sale prices against observed sale prices.

## Modeling Notes

The target variable is `SalePrice`. The notebook uses a 70/30 train-test split with `random_state=42`, and scaling parameters are fit on the training data before being applied to the test data.

The one-hot encoding is implemented explicitly so that the effect of categorical expansion can be inspected. The notebook is intended as an exploratory modeling study; it does not persist a fitted model or provide a production inference pipeline.

## Running the Notebook

Open `notebook.ipynb` in VS Code or Jupyter with a Python environment containing pandas, NumPy, scikit-learn, Matplotlib, and Seaborn. Run the cells from top to bottom from the project directory so the relative CSV path resolves correctly.
