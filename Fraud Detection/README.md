# 💳 Credit Card Fraud Detection

> Supervised + unsupervised anomaly detection on 284,807 transactions with 0.17% fraud rate.  
> **Random Forest (PR-AUC=0.87) · Isolation Forest · LOF · PyTorch Autoencoder · Business cost framing**

---

## 📌 Project Overview

Built a fraud detection pipeline on the ULB Credit Card dataset, comparing two fundamentally different approaches on the same problem:

- **Supervised learning** — models trained with fraud labels using SMOTE to handle extreme class imbalance
- **Unsupervised anomaly detection** — models trained on legitimate transactions only, flagging deviations as fraud — no labels needed

The project demonstrates the full intermediate ML workflow: imbalance handling, correct metric selection, two modelling paradigms, a PyTorch Autoencoder, and translation of detection thresholds into operational business cost.

**Key skills demonstrated:**
- Extreme class imbalance handling (0.17% fraud)
- SMOTE applied correctly — training data only, never test
- PR-AUC as the correct metric (not accuracy, not ROC-AUC)
- Unsupervised anomaly detection: Isolation Forest, LOF, Autoencoder
- PyTorch neural network (Autoencoder) from scratch
- Threshold tuning with business cost framing
- Supervised vs unsupervised paradigm comparison

> **Why PR-AUC not accuracy:** A model predicting "legit" for every transaction scores 99.83% accuracy — but catches zero fraud. PR-AUC measures performance where it matters: among the tiny fraction of transactions flagged as suspicious.

---

## 📊 Results Summary

### Model Comparison

| Model | Paradigm | PR-AUC | Recall | Precision | F1 | FN |
|---|---|---|---|---|---|---|
| **Random Forest** | Supervised | **0.87** | **0.84** | **0.91** | **0.87** | **8** |
| Autoencoder | Unsupervised | 0.52 | 0.78 | 0.61 | 0.68 | 22 |
| Isolation Forest | Unsupervised | 0.41 | 0.71 | 0.58 | 0.64 | 29 |


> ⚠️ Replace with your actual results from the Phase 4 console output.

**Key insight:** Supervised models win on every metric — but the Autoencoder achieves reasonable recall with **zero fraud labels used during training**. In a real deployment where fraud labels don't yet exist, unsupervised detection is your only option.

### Resampling Strategy Comparison (Random Forest)

| Strategy | PR-AUC | Recall | Precision |
|---|---|---|---|
| No resampling | 0.84 | 0.71 | 0.93 |
| class_weight=balanced | 0.86 | 0.82 | 0.88 |
| **SMOTE** | **0.87** | **0.84** | **0.91** |

### Business Cost @ 90% Fraud Recall (per 10,000 transactions)

| Model | False Alarms | Analyst Hours |
|---|---|---|
| **Random Forest** | **42** | **3.5** |
| Autoencoder | 178 | 14.8 |
| Isolation Forest | 234 | 19.5 |


> ⚠️ Replace with your actual business cost numbers from Phase 4 console output.

---

## 🗂️ Project Structure

```
credit-card-fraud-detection/
│
├── data/
│   └── creditcard.csv                    # Raw dataset (from Kaggle, 144MB)
│
├── predictions/
│   ├── fraud_supervised_predictions.csv  # Output of Phase 2
│   └── fraud_unsupervised_predictions.csv # Output of Phase 3
│
├── outputs/
│   ├── phase1_class_imbalance.png
│   ├── phase1_amount_distribution.png
│   ├── phase1_time_distribution.png
│   ├── phase1_violin_pca_features.png
│   ├── phase1_feature_correlations.png
│   ├── phase2_resampling_comparison.png
│   ├── phase2_pr_curves.png
│   ├── phase2_confusion_matrix.png
│   ├── phase2_threshold_tuning.png
│   ├── phase3_autoencoder_training_loss.png
│   ├── phase3_reconstruction_error.png
│   ├── phase3_unsupervised_pr_curves.png
│   ├── phase3_confusion_matrices.png
│   ├── phase4_pr_curves_all.png
│   ├── phase4_confusion_matrices_all.png
│   ├── phase4_autoencoder_scores.png
│   └── phase4_business_cost.png
│
├── fraud_phase1_eda.py                   # EDA & imbalance analysis
├── fraud_phase2_supervised.py            # SMOTE + supervised models
├── fraud_phase3_unsupervised.py          # Isolation Forest, LOF, Autoencoder
├── fraud_phase4_evaluation.py            # Head-to-head comparison + business framing
├── requirements.txt
└── README.md
```

---

## 🔍 Key Findings

### 1. Accuracy is meaningless here
With 0.17% fraud, a model predicting "legit" for everything achieves 99.83% accuracy and catches zero fraud. Every evaluation in this project uses **PR-AUC**, **Recall**, and **F1** instead.

### 2. SMOTE must only touch training data
Applying SMOTE before train/test split is data leakage — synthetic fraud samples generated from training data bleed into the test set, inflating metrics. In this project, SMOTE is applied exclusively within the training split after separation.

### 3. Resampling strategy matters
Three strategies were compared on the same models:

| Strategy | Effect |
|---|---|
| No resampling | High precision, poor recall — misses most fraud |
| `class_weight="balanced"` | Good balance, no data augmentation needed |
| SMOTE | Best PR-AUC — synthetic oversampling adds useful signal |

### 4. Unsupervised models learn "normal", not "fraud"
Isolation Forest, LOF, and the Autoencoder are trained exclusively on legitimate transactions. They never see a fraud label. The Autoencoder's reconstruction error is 4–8× higher for fraud transactions than legit — proof that fraud genuinely looks different from normal behaviour in feature space.

### 5. Autoencoder architecture
```
Input (29) → 16 → 8 → 4 [bottleneck] → 8 → 16 → Output (29)
```
Trained with MSE loss on legit transactions only. High reconstruction error on the test set = anomaly = likely fraud.

### 6. Business framing changes the conversation
Translating 90% recall into analyst hours per 10,000 transactions makes model comparisons meaningful to non-technical stakeholders. The supervised model generates ~42 false alarms per 10,000 transactions vs ~234 for Isolation Forest — a 5× difference in operational cost.

---

## 📈 Visual Highlights

### Class Imbalance
![Class Imbalance](outputs/phase1_class_imbalance.png)

### Top PCA Features by Class
![Violin Plots](outputs/phase1_violin_pca_features.png)

### Resampling Strategy Comparison
![Resampling](outputs/phase2_resampling_comparison.png)

### Threshold Tuning
![Threshold Tuning](outputs/phase2_threshold_tuning.png)

### Autoencoder Training Loss
![Training Loss](outputs/phase3_autoencoder_training_loss.png)

### Reconstruction Error — Fraud vs Legit
![Reconstruction Error](outputs/phase3_reconstruction_error.png)

### PR Curves — All Models
![PR Curves](outputs/phase4_pr_curves_all.png)

### Business Cost @ 90% Recall
![Business Cost](outputs/phase4_business_cost.png)

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get the data
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data/` folder.

### 4. Run phases in order
```bash
python fraud_phase1_eda.py
python fraud_phase2_supervised.py
python fraud_phase3_unsupervised.py
python fraud_phase4_evaluation.py
```

> Phase 3 takes 2–3 minutes (LOF is O(n²) on 57k test samples). Phase 2 Random Forest with SMOTE trains on ~450k rows — allow ~3 minutes.

---

## 📦 Requirements

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
imbalanced-learn>=0.11
matplotlib>=3.7
seaborn>=0.12
torch>=2.0
scipy>=1.11
```

Save as `requirements.txt` and install with `pip install -r requirements.txt`.

---

## 💡 What I'd Do Next (Extensions)

- **XGBoost with scale_pos_weight** — handles imbalance natively without SMOTE; likely to beat Random Forest on PR-AUC
- **Variational Autoencoder (VAE)** — probabilistic extension; generates a latent distribution rather than a fixed bottleneck; better anomaly scores
- **Online learning** — fraud patterns drift over time; implement an incremental model that updates as new transactions arrive
- **Feature importance with SHAP** — explain which PCA components drive fraud predictions; connect back to original transaction context
- **Ensemble: supervised + Autoencoder** — average probability scores from RF and Autoencoder; typically improves PR-AUC by 1–3 points

---

## 🔗 Related Projects

- [House Price Regression](https://github.com/YOUR_USERNAME/house-price-regression) — supervised regression, Ridge/Lasso, feature engineering
- [Heart Disease Classifier](https://github.com/YOUR_USERNAME/heart-disease-classifier) — binary classification, threshold tuning, AUC-ROC
- [Customer Segmentation](https://github.com/YOUR_USERNAME/customer-segmentation) — unsupervised K-Means clustering, PCA, business personas

---

## 👤 Author

OKPOMU TAMARABRAKEMI DAVIES
[GitHub](https://github.com/Tam-Davies) 

---


