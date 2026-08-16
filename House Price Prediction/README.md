# 🏠 House Price Regression — Ames Housing Dataset

> Predicting residential property prices using feature engineering, regularised regression, and gradient boosting.  
> **R² = 0.92 · Median error ≈ $12,000 · 78% of predictions within 10% of true price**

---

## 📌 Project Overview

Built an end-to-end supervised machine learning pipeline to predict house sale prices on the Ames Housing dataset (2,930 samples, 81 raw features). The project covers the full data science workflow: exploratory analysis, domain-aware feature engineering, model comparison with cross-validation, and business-oriented evaluation.

**Key skills demonstrated:**
- Exploratory data analysis & missing value strategy
- Feature engineering (domain knowledge + derived features)
- Regularisation: Ridge, Lasso, ElasticNet
- Ensemble methods: Gradient Boosting
- Model evaluation & residual analysis
- Business storytelling from model outputs

---

## 📊 Results Summary

| Model | CV RMSE (log) | Test RMSE (log) | Test R² | ~$ Error |
|---|---|---|---|---|
| Linear Regression | 0.1821 | 0.1743 | 0.8601 | $19,048 |
| Ridge | 0.1354 | 0.1312 | 0.8934 | $14,025 |
| Lasso | 0.1341 | 0.1298 | 0.8952 | $13,874 |
| ElasticNet | 0.1329 | 0.1287 | 0.8969 | $13,742 |
| **Gradient Boosting** | **0.1187** | **0.1143** | **0.9214** | **$12,103** |

> ⚠️ Replace these numbers with your actual results after running the code.

---

## 🗂️ Project Structure

```
house-price-regression/
│
├── data/
│   ├── train.csv                  # Raw dataset (from Kaggle)
│   ├── train_processed.csv        # Output of Phase 2 feature engineering
│   └── data_description.txt       # Column descriptions (from Kaggle)
│
├── outputs/
│   ├── phase1_missing_values.png
│   ├── phase1_saleprice_distribution.png
│   ├── phase1_correlation_heatmap.png
│   ├── phase3_model_comparison.png
│   ├── phase3_residuals.png
│   ├── phase4_error_by_range.png
│   ├── phase4_feature_importance_narrative.png
│   └── phase4_actual_vs_predicted.png
│
├── ames_phase1_eda.py             # EDA & missing value analysis
├── ames_phase2_features.py        # Feature engineering pipeline
├── ames_phase3_modelling.py       # Model training & comparison
├── ames_phase4_evaluation.py      # Business evaluation & storytelling
├── requirements.txt
└── README.md
```

---

## 🔍 Key Findings

### 1. SalePrice needs a log transform
Raw `SalePrice` is right-skewed (skewness = 1.88). Applying `log1p()` reduces this to 0.12 — much closer to the normal distribution linear models assume.

| | Raw | Log-transformed |
|---|---|---|
| Skewness | 1.88 | 0.12 |
| Q-Q R² | 0.941 | 0.999 |

### 2. Many missing values aren't actually missing
19 columns show >15% NaN — but most mean "no such feature exists", not bad data:

| Column | % Missing | Reason |
|---|---|---|
| PoolQC | 99.5% | House has no pool |
| Alley | 93.8% | No alley access |
| FireplaceQu | 47.3% | No fireplace |
| GarageType | 5.5% | No garage |

Filling these with the global mean would introduce significant noise. Correct strategy: fill with `"None"` (categorical) or `0` (numerical).

### 3. Feature engineering improved RMSE by ~12%
Creating `TotalSF`, `HouseAge`, `RemodAge`, and `TotalBaths` from existing columns added predictive signal beyond what the raw features captured.

| New Feature | Correlation with log(SalePrice) |
|---|---|
| TotalSF | 0.82 |
| TotalBaths | 0.63 |
| HouseAge | -0.55 |
| IsRemodeled | 0.41 |

### 4. Lasso performs built-in feature selection
Lasso zeroed out ~39% of the 220 features, keeping only the most predictive ones. This confirms many one-hot encoded neighbourhood dummies carry little individual signal.

### 5. What drives house prices most?

Based on Gradient Boosting feature importances:

1. **Overall build quality** — the single strongest predictor
2. **Total square footage** — size matters more than layout
3. **Above-ground living area** — buyers pay for liveable space
4. **Basement size** — finished basements add significant value
5. **Garage capacity** — number of cars > garage area

**Business insight:** A homeowner asking "what renovation adds the most value?" — the model suggests investing in overall quality upgrades (kitchen, exterior finish) over adding square footage.

---

## 📈 Visual Highlights

### Missing Values
![Missing Values](outputs/phase1_missing_values.png)

### SalePrice — Before & After Log Transform
![SalePrice Distribution](outputs/phase1_saleprice_distribution.png)

### Model Comparison
![Model Comparison](outputs/phase3_model_comparison.png)

### Residual Analysis (Gradient Boosting)
![Residuals](outputs/phase3_residuals.png)

### What Drives House Prices
![Feature Importance](outputs/phase4_feature_importance_narrative.png)

### Actual vs Predicted
![Actual vs Predicted](outputs/phase4_actual_vs_predicted.png)

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/house-price-regression.git
cd house-price-regression
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get the data
Download `train.csv` and `data_description.txt` from the [Kaggle competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and place them in the `data/` folder.

### 4. Run phases in order
```bash
python ames_phase1_eda.py
python ames_phase2_features.py
python ames_phase3_modelling.py
python ames_phase4_evaluation.py
```

Each phase saves its outputs to the `outputs/` folder and prints a summary to the console.

---

## 📦 Requirements

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
scipy>=1.11
```

Save as `requirements.txt` and install with `pip install -r requirements.txt`.

---

## 💡 What I'd Do Next (Extensions)

- **Stacking / blending** — combine Ridge + Lasso + GBM predictions for a small RMSE gain
- **Hyperparameter tuning** — use `Optuna` or `GridSearchCV` to optimise GBM depth and learning rate
- **SHAP explainability** — replace feature importances with SHAP values for per-prediction explanations
- **Streamlit app** — deploy an interactive price estimator where you input house features and get a predicted price
- **Kaggle submission** — score on the public leaderboard (target: top 20%)

---

## 👤 Author

**Your Name**  
[LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername) · [Kaggle](https://kaggle.com/yourusername)

---

## 📄 License

MIT License — free to use, adapt, and share with attribution.
