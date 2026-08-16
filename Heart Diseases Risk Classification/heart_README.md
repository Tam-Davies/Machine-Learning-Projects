# ❤️ Heart Disease Risk Classifier

> Binary classification of heart disease risk using the Cleveland dataset.  
> **Best model: Logistic Regression · AUC-ROC = 0.9065 · Recall = 0.9286 · KNN & SVM achieved lowest FN = 1 (Recall = 0.9643)**

---

## 📌 Project Overview

Built an end-to-end binary classification pipeline to predict heart disease presence in patients using the Cleveland Heart Disease dataset (303 patients, 13 clinical features). The project covers the full data science workflow: exploratory analysis, preprocessing with a scikit-learn Pipeline, comparison of five classification models, and clinical evaluation with threshold tuning.

**Key skills demonstrated:**
- Binary classification with medical domain context
- Preprocessing with `ColumnTransformer` + `Pipeline` (prevents data leakage)
- `StratifiedKFold` cross-validation for imbalanced classes
- Classification metric literacy — F1, AUC-ROC, Precision, Recall
- Threshold tuning to minimise false negatives (clinical priority)
- Decision Tree visualisation & Random Forest feature importance

> **Why this matters beyond accuracy:** A model predicting "no disease" for every patient gets 46% accuracy — but misses every sick patient. This project shows why F1, Recall, and AUC-ROC are the metrics that count in medical classification.

---

## 📊 Results Summary

| Model | Accuracy | F1 | Precision | Recall | AUC-ROC |
|---|---|---|---|---|---|
| **Logistic Regression** ✓ | **0.8429** | **0.8223** | **0.8589** | 0.7925 | **0.9065** |
| KNN | 0.8014 | 0.7798 | 0.7969 | 0.7652 | 0.8628 |
| Decision Tree | 0.7315 | 0.6955 | 0.7242 | 0.6743 | 0.7903 |
| Random Forest | 0.8139 | 0.7893 | 0.8264 | 0.7648 | 0.8926 |
| SVM (RBF) | 0.8180 | 0.7908 | 0.8378 | 0.7561 | 0.8866 |

> Best model by AUC-ROC: **Logistic Regression (0.9065)**. Note that KNN and SVM achieved the fewest false negatives (FN=1) on the test set — an important clinical consideration separate from overall AUC.

### False Negatives per Model (Test Set)

| Model | False Negatives | Recall |
|---|---|---|
| KNN | **1** | **0.9643** |
| SVM | **1** | **0.9643** |
| Logistic Regression | 2 | 0.9286 |
| Random Forest | 2 | 0.9286 |
| Decision Tree | 5 | 0.8214 |

> In a clinical screening context, KNN and SVM are the strongest performers — only 1 missed disease case each on the test set, despite Logistic Regression leading on AUC-ROC. This highlights why no single metric tells the whole story.

---

## 🗂️ Project Structure

```
heart-disease-classifier/
│
├── data/
│   └── heart.csv                   # Cleveland dataset (from Kaggle or UCI)
│
├── outputs/
│   ├── phase1_class_balance.png
│   ├── phase1_violin_plots.png
│   ├── phase1_correlation_heatmap.png
│   ├── phase2_3_model_comparison.png
│   ├── phase2_3_knn_k_search.png
│   ├── phase2_3_decision_tree.png
│   ├── phase4_confusion_matrices.png
│   ├── phase4_roc_curves.png
│   ├── phase4_pr_curve.png
│   ├── phase4_threshold_tuning.png
│   └── phase4_feature_importance.png
│
├── heart_phase1_eda.py             # EDA, class balance, violin plots
├── heart_phase2_3_models.py        # Preprocessing pipeline + 5 models
├── heart_phase4_evaluation.py      # Confusion matrices, ROC/PR, threshold tuning
├── requirements.txt
└── README.md
```

---

## 🔍 Key Findings

### 1. Accuracy is the wrong metric here
With a 54/46 class split, a dummy classifier predicting "disease" for everyone scores 54% accuracy. Recall and AUC-ROC tell the real story — a missed diagnosis (false negative) is far more costly than a false alarm.

### 2. Max heart rate is the strongest separator
`thalach` (maximum heart rate achieved) shows the clearest visual separation between classes in the violin plots. Disease patients have a notably lower median max heart rate — consistent with clinical literature.

| Feature | No Disease (median) | Disease (median) | Separation |
|---|---|---|---|
| Max heart rate (thalach) | 158 | 141 | 10.8% |
| ST depression (oldpeak) | 0.0 | 1.5 | strong |
| Major vessels (ca) | 0.0 | 1.0 | strong |
| Age | 52 | 57 | 9.6% |

### 3. StratifiedKFold matters
Standard KFold can produce folds with 70/30 class ratios by chance. `StratifiedKFold` preserves the 54/46 split across every fold — essential for reliable CV scores on classification tasks.

### 4. Decision Tree is fully explainable
The trained Decision Tree (max_depth=4) splits first on `ca` (major vessels) or `cp` (chest pain type) — both clinically validated. Every prediction can be traced back to a readable rule, making it the most explainable model in the comparison.

### 5. No single metric picks the best model
Logistic Regression led on AUC-ROC (0.9065), but KNN and SVM each missed only 1 disease case (Recall=0.9643) — fewer than Logistic Regression (FN=2). In a screening context where minimising missed diagnoses is the priority, KNN or SVM would be the clinical choice despite a lower AUC. This tradeoff is the central evaluation story of the project.

### 6. Top clinical risk factors (Random Forest)
Feature importances align with established cardiology research:

1. **Major vessels (ca)** — number of major vessels coloured by fluoroscopy
2. **Max heart rate (thalach)** — lower max HR strongly associated with disease
3. **ST depression (oldpeak)** — exercise-induced ST depression
4. **Chest pain type (cp)** — asymptomatic chest pain is paradoxically high-risk
5. **Age** — risk increases with age

---

## 📈 Visual Highlights

### Class Balance
![Class Balance](outputs/phase1_class_balance.png)

### Feature Distributions by Class
![Violin Plots](outputs/phase1_violin_plots.png)

### Model Comparison
![Model Comparison](outputs/phase2_3_model_comparison.png)

### Decision Tree (max_depth=4)
![Decision Tree](outputs/phase2_3_decision_tree.png)

### Confusion Matrices — All Models
![Confusion Matrices](outputs/phase4_confusion_matrices.png)

### ROC Curves
![ROC Curves](outputs/phase4_roc_curves.png)

### Threshold Tuning
![Threshold Tuning](outputs/phase4_threshold_tuning.png)

### Feature Importance
![Feature Importance](outputs/phase4_feature_importance.png)

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-classifier.git
cd heart-disease-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get the data
Download `heart.csv` from [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) or [UCI](https://archive.ics.uci.edu/dataset/45/heart+disease) and place it in the `data/` folder.

### 4. Run phases in order
```bash
python heart_phase1_eda.py
python heart_phase2_3_models.py
python heart_phase4_evaluation.py
```

---

## 📦 Requirements

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
```

Save as `requirements.txt` and install with `pip install -r requirements.txt`.

---

## 💡 What I'd Do Next (Extensions)

- **SHAP values** — replace feature importances with per-prediction SHAP explanations; show which features pushed each patient toward or away from a positive diagnosis
- **Hyperparameter tuning** — `GridSearchCV` on Random Forest (`n_estimators`, `max_depth`, `min_samples_split`) for a potential AUC gain
- **Calibration** — use `CalibratedClassifierCV` to ensure predicted probabilities are well-calibrated; important when using probabilities for clinical risk scoring
- **Streamlit app** — deploy an interactive risk calculator where a clinician inputs patient vitals and gets a probability score with the top contributing factors
- **Compare with XGBoost** — add a gradient boosting model to the comparison; likely to edge out Random Forest on this dataset

---

## 🔗 Related Projects

- [House Price Regression](https://github.com/YOUR_USERNAME/house-price-regression) — supervised regression, Ridge/Lasso regularisation, feature engineering on the Ames Housing dataset

---

## 👤 Author

**Your Name**  
[LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername) · [Kaggle](https://kaggle.com/yourusername)

---

## 📄 License

MIT License — free to use, adapt, and share with attribution.
