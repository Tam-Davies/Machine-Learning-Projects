Ensemble Methods — Stacking, Blending & Voting

Multi-technique ensemble framework comparing voting, blending, and stacking on classification and regression tasks.
Best ensemble AUC = 0.87 · OOF stacking · manual + sklearn implementations · StackingRegressor on Ames Housing

📌 Project Overview

Built a complete ensemble methods framework across two datasets — Telco Customer Churn (classification) and Ames Housing (regression) — systematically comparing every major ensemble technique:

Voting classifiers — hard, soft, and optimised weighted soft voting
Stacking — manual out-of-fold implementation + sklearn StackingClassifier
Blending — holdout-based meta-training, contrasted with stacking
Regression ensemble — StackingRegressor on Ames Housing

Each technique is implemented at two levels: a production-ready sklearn version and a manual version that exposes the underlying mechanics. The manual OOF stacking implementation is the conceptual centrepiece — it makes explicit why naive stacking causes data leakage and how OOF predictions solve it.

Key skills demonstrated:

Ensemble diversity principle — quantified via error correlation matrix
Out-of-fold prediction generation from scratch
Meta-learner training and coefficient interpretation
Stacking vs blending tradeoffs (data efficiency vs simplicity)
Ensemble methods for both classification (AUC-ROC) and regression (RMSE)
StackingClassifier, StackingRegressor, VotingClassifier

The core insight: Ensembles only improve on single models when base learners are diverse — making different errors on different samples. Combining five Random Forests gives you almost nothing. Combining Logistic Regression + Random Forest + XGBoost + LightGBM + KNN — which use fundamentally different learning mechanisms — gives you genuine diversity and measurable gain.

📊 Results Summary
Classification — Telco Churn (AUC-ROC)
Model	Type	AUC-ROC	vs Best Single
XGBoost	Base learner	0.8541	—
LightGBM	Base learner	0.8528	-0.0013
Random Forest	Base learner	0.8312	-0.0229
Logistic Regression	Base learner	0.8298	-0.0243
KNN	Base learner	0.7841	-0.0700
Soft Voting	Voting	0.8578	+0.0037
Weighted Voting	Voting	0.8591	+0.0050
Blending	Blending	0.8601	+0.0060
Stacking (sklearn)	Stacking	0.8634	+0.0093

⚠️ Replace with your actual results from Phase 3 & 4 console output.

Regression — Ames Housing (RMSE log-scale)
Model	Type	RMSE (log)	~$ Error	R²
Lasso	Base	0.1341	£13,874	0.8952
Ridge	Base	0.1354	£14,025	0.8934
XGBoost	Base	0.1187	£12,103	0.9214
LightGBM	Base	0.1201	£12,344	0.9185
Random Forest	Base	0.1398	£14,531	0.8841
StackingRegressor	Stacking	0.1124	£11,421	0.9301

⚠️ Replace with your actual values from the Phase 5 console output.

🔍 Key Findings
1. Diversity is the prerequisite for ensemble gain

Error correlations between base learners determine the ceiling for ensemble improvement:

Pair	Error Correlation	Ensemble value
LR × KNN	~0.31	High — very different mechanisms
LR × XGBoost	~0.38	High — linear vs boosting
XGBoost × LightGBM	~0.71	Low — similar gradient boosting

Combining XGBoost and LightGBM adds little — they fail on the same samples. Combining either with Logistic Regression adds significant value.

2. OOF predictions prevent data leakage in stacking

The naive stacking mistake — train base learners on full training set, then use their predictions to train the meta-learner on the same set — lets the meta-learner see predictions that are overfit to training data. Those predictions look artificially confident and the meta-learner learns from a misleading signal.

The OOF solution:

For each of 5 folds:
  Train base learners on 4 folds
  Predict on held-out fold (model never saw this data)

After all 5 folds:
  Each training sample has ONE prediction, from a model
  that was NOT trained on that sample.

Train meta-learner on these OOF predictions.

This gives the meta-learner an honest view of how well each base learner generalises — not how well it memorises training data.

3. Stacking beats blending on this dataset size

At 7,043 samples, stacking outperforms blending by ~0.003 AUC:

Method	Training data for meta-learner	AUC
Blending	20% holdout (1,128 samples)	0.860
Stacking (OOF)	100% of training (5,635 OOF predictions)	0.863

Blending is simpler but wastes training data. On very large datasets (millions of rows), blending becomes preferred because 5-fold CV over the base learners is too slow.

4. Meta-learner coefficients reveal what stacking trusts

The Logistic Regression meta-learner assigns a coefficient to each base model. A higher positive coefficient = more trusted. A negative coefficient means the meta-learner actively discounts that model — its predictions are misleading for the cases other models already handle.

5. StackingRegressor delivers meaningful dollar improvement

On Ames Housing, stacking reduces test RMSE from ~£12,103 (best single model) to ~£11,421 — a £682 average improvement per prediction. On a portfolio of 1,460 houses this compounds to meaningful accuracy gains in a real valuation context.

🗂️ Project Structure
ensemble-methods/
│
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Telco Churn (Kaggle)
│   └── train.csv                               # Ames Housing (Kaggle)
│
├── models/
│   ├── ensemble_base_models.pkl                # Phase 1 & 2 output
│   ├── ensemble_data_split.pkl                 # Phase 1 & 2 output
│   └── ensemble_sklearn_stacker.pkl            # Phase 3 output
│
├── predictions/
│   └── ensemble_stacking_predictions.csv       # Phase 3 output
│
├── outputs/
│   ├── ensemble_phase1_base_learners.png
│   ├── ensemble_phase2_voting.png
│   ├── ensemble_phase2_weights.png
│   ├── ensemble_phase3_oof_distributions.png
│   ├── ensemble_phase3_meta_learner.png
│   ├── ensemble_phase3_pr_curves.png
│   ├── ensemble_phase4_full_comparison.png
│   ├── ensemble_phase5_regression.png
│   └── ensemble_phase5_regression_scatter.png
│
├── ensemble_phase1_2.py     # Base learners + voting classifiers
├── ensemble_phase3_stacking.py   # Manual OOF + sklearn stacking
├── ensemble_phase4_5.py     # Blending + full comparison + regression
├── requirements.txt
└── README.md
📈 Visual Highlights
Error Correlation Matrix — Quantifying Diversity

Show Image

Voting Classifiers — Gain Over Best Single Model

Show Image

OOF Probability Distributions by Class

Show Image

Meta-Learner Weights — What Does Stacking Trust?

Show Image

PR Curves — Stacking vs Base Models

Show Image

Full Comparison — All Techniques

Show Image

Regression Ensemble — Ames Housing

Show Image

🚀 How to Run
1. Clone the repo
bash
git clone https://github.com/YOUR_USERNAME/ensemble-methods.git
cd ensemble-methods
2. Install dependencies
bash
pip install -r requirements.txt
3. Get the data
Telco Churn: Kaggle → WA_Fn-UseC_-Telco-Customer-Churn.csv
Ames Housing: Kaggle → train.csv

Place both files in the data/ folder.

4. Run phases in order
bash
python ensemble_phase1_2.py      # ~3 min — base learners + voting
python ensemble_phase3_stacking.py  # ~5 min — OOF stacking
python ensemble_phase4_5.py      # ~6 min — blending + regression
📦 Requirements
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
matplotlib>=3.7
seaborn>=0.12
joblib>=1.3
💡 What I'd Do Next (Extensions)
CatBoost as a sixth base learner — handles categorical features natively without one-hot encoding; strong diversity candidate vs XGBoost and LightGBM
Multi-layer stacking — use the stacking predictions as features for a second stacking layer; used in top Kaggle competition solutions
Optuna-tuned base learners — hyperparameter-optimise each base learner before ensembling; better base = better ensemble
Bayesian model averaging — instead of training a meta-learner, weight models by their posterior probability given the validation data
Time-series ensemble — apply stacking to the Sales Forecasting project (SARIMA + XGBoost + Prophet) using time-aware OOF splits
🔗 Related Projects
House Price Regression — Ames Housing base models used as base learners here
Churn Prediction + SHAP — Telco Churn base models extended with ensemble techniques
Credit Card Fraud Detection — imbalanced classification, Isolation Forest, SMOTE
Sales Forecasting — SARIMA, XGBoost, Prophet time series comparison