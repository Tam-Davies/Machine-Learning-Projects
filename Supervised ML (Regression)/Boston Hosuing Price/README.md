# Boston Housing Price Regression

This project explores transformations of the Boston housing price target and fits a polynomial-feature linear regression model.

## Project Files

- [`notebook.ipynb`](notebook.ipynb): Complete exploratory analysis and model-building workflow.
- `data/boston_housing_clean.pickle`: Cleaned dataset and description used by the notebook.

## Workflow

1. Load the cleaned Boston housing data and identify `MEDV` as the target.
2. Inspect the target distribution with a histogram and D'Agostino's normality test.
3. Compare log and square-root transformations.
4. Use a Box-Cox transformation to find a better-behaved target.
5. Separate predictors from `MEDV` and create degree-2 polynomial features.
6. Split the data into training and test sets with 30% reserved for testing.
7. Standardize features using training data only.
8. Transform the training target with Box-Cox and fit linear regression.
9. Generate test predictions and convert values back to the original price scale.

## Results

- Log transformation normality test: statistic `17.2180`, p-value `0.000182`.
- Square-root transformation normality test: statistic `20.4871`, p-value `0.0000356`.
- Box-Cox lambda: `0.2166`.
- Box-Cox normality test: statistic `4.5135`, p-value `0.1047`.
- Training target shape after transformation: `(354,)`.
- The inverse Box-Cox sanity check reproduces the first ten observed prices.

The notebook currently fits the model and generates predictions, but it does not yet report test-set R2, MAE, or RMSE. Those metrics should be added before comparing model quality or using the model for decisions.

## Running the Notebook

Open `notebook.ipynb` in VS Code with the Jupyter and Python extensions installed. Run the cells from top to bottom with a compatible Python environment containing NumPy, pandas, Matplotlib, SciPy, and scikit-learn.
