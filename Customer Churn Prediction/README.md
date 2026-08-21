📉 Customer Churn Prediction + SHAP Explainability

XGBoost churn classifier with per-customer SHAP explanations and business cost optimisation.
AUC-ROC = 0.85 · 3 risk tiers · £X net monthly benefit · intervention table for retention team

📌 Project Overview

Built an end-to-end churn prediction pipeline on the IBM Telco Customer Churn dataset (7,043 customers, 21 features), combining a high-performing XGBoost classifier with SHAP (SHapley Additive exPlanations) to explain every prediction at the individual customer level.

The project goes beyond model accuracy — it translates predictions into a prioritised retention intervention table, optimises the decision threshold by business cost rather than F1, and estimates monthly revenue retained by deploying the model vs doing nothing.

Key skills demonstrated:

Binary classification with class imbalance (scale_pos_weight)
ColumnTransformer + Pipeline preprocessing
StratifiedKFold model comparison across 4 classifiers
SHAP TreeExplainer — global and per-customer explainability
Threshold optimisation by net business benefit (not just F1)
Risk tier segmentation and retention intervention table
Revenue framing — CLV, offer cost, model ROI

What makes this different from a standard churn model: Anyone can train XGBoost on this dataset and get AUC ≈ 0.85. The SHAP layer answers the question a retention manager actually cares about: "Why is this specific customer at risk, and what should we do about it?" That explanation — not the AUC score — is the business value.

📊 Results Summary
Model Comparison (5-Fold Stratified CV)
Model	AUC-ROC	F1	Precision	Recall
Logistic Regression	0.840	0.612	0.648	0.580
Gradient Boosting	0.845	0.621	0.655	0.591
Random Forest	0.838	0.608	0.641	0.578
XGBoost	0.854	0.638	0.667	0.611

⚠️ Replace with your actual CV scores from Phase 2 console output.

Risk Tier Segmentation
Tier	Customers	Actual Churn Rate	Action
🔴 High Risk (≥70%)	~185	~65%	Immediate retention offer
🟠 Medium Risk (40–70%)	~210	~32%	Proactive outreach
🟢 Low Risk (<40%)	~1,014	~5%	Standard engagement

⚠️ Replace counts and churn rates from your Phase 4 console output.

Business Impact (per month, test set)
Scenario	Revenue Saved	Cost	Net Benefit
No model	£0	£0	£0
Model (threshold=0.35)	£X	£Y	£Z

⚠️ Fill in from Phase 4 console — net benefit and ROI figures.

🗂️ Project Structure
churn-prediction-shap/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset (Kaggle)
│
├── predictions/
│   ├── churn_test_predictions.csv              # Phase 2 output
│   ├── churn_train_preprocessed.csv            # Phase 2 output
│   ├── churn_intervention_table.csv            # Phase 3 output
│   └── churn_full_dashboard.csv                # Phase 4 output
│
├── outputs/
│   ├── phase1_churn_rate.png
│   ├── phase1_churn_by_contract.png
│   ├── phase1_churn_by_tenure.png
│   ├── phase1_numeric_violins.png
│   ├── phase1_correlation_heatmap.png
│   ├── phase2_cv_comparison.png
│   ├── phase2_roc_pr_curves.png
│   ├── phase2_confusion_matrices.png
│   ├── phase2_threshold_tuning.png
│   ├── phase3_shap_summary.png
│   ├── phase3_shap_bar.png
│   ├── phase3_shap_waterfall.png
│   ├── phase3_shap_dependence.png
│   ├── phase3_shap_force.png
│   ├── phase3_shap_segment.png
│   ├── phase4_risk_tiers.png
│   ├── phase4_tier_profiles.png
│   ├── phase4_threshold_business.png
│   └── phase4_revenue_impact.png
│
├── churn_phase1_eda.py          # EDA & churn rate analysis
├── churn_phase2_models.py       # Preprocessing pipeline + model comparison
├── churn_phase3_shap.py         # SHAP explainability — all 5 chart types
├── churn_phase4_business.py     # Risk tiers, threshold tuning, revenue model
├── requirements.txt
└── README.md
🔍 Key Findings
1. Contract type is the dominant churn predictor

Month-to-month customers churn at ~42% vs ~11% (one-year) and ~3% (two-year). The clearest retention lever: incentivise contract upgrades for high-risk month-to-month customers.

Contract	Churn Rate
Month-to-month	~42%
One year	~11%
Two year	~3%
2. Early tenure is the critical retention window

Customers in their first 6 months churn at ~50%. After 2 years, churn drops below 10%. The model correctly identifies new customers as highest risk — the SHAP dependence plot for tenure shows a steep drop in churn risk between months 0 and 18.

3. SHAP replaces feature importance

Standard tree feature importance tells you which features matter globally. SHAP tells you:

Which features matter globally (bar plot, beeswarm)
Which direction each feature pushes predictions (beeswarm colour)
Exactly why this specific customer is at risk (waterfall, force plot)
Non-linear relationships between feature value and risk (dependence plot)
4. Threshold optimisation changes the decision

At the default threshold of 0.50, the model minimises neither business cost nor F1. The optimal threshold for net revenue benefit is typically 0.30–0.40 — flagging more customers for outreach at the cost of more false alarms, which is the right tradeoff when CLV (£2,080) >> offer cost (£50).

5. Model ROI is concrete and defensible

At optimal threshold:

Retention offers sent to tp + fp customers
tp × 40% churners actually retained
Net benefit = revenue retained − offer cost
ROI typically 300–600% — a model that pays for itself many times over
📈 Visual Highlights
Churn by Contract Type

Show Image

Churn by Tenure

Show Image

Model Comparison — CV Metrics

Show Image

ROC + Precision-Recall Curves

Show Image

SHAP Summary Plot (Beeswarm)

Show Image

SHAP Waterfall — Individual Customer Breakdowns

Show Image

SHAP Dependence — Tenure & Monthly Charges

Show Image

Risk Tier Segmentation

Show Image

Threshold Tuning — Business Cost

Show Image

Revenue Impact — Model vs No Model

Show Image

🚀 How to Run
1. Clone the repo
bash
git clone https://github.com/YOUR_USERNAME/churn-prediction-shap.git
cd churn-prediction-shap
2. Install dependencies
bash
pip install -r requirements.txt
3. Get the data

Download WA_Fn-UseC_-Telco-Customer-Churn.csv from Kaggle and place it in the data/ folder.

4. Run phases in order
bash
python churn_phase1_eda.py
python churn_phase2_models.py
python churn_phase3_shap.py
python churn_phase4_business.py

Each script saves outputs to the outputs/ folder and prints a summary to the console.

📦 Requirements
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
shap>=0.44
matplotlib>=3.7
seaborn>=0.12

Save as requirements.txt and install with pip install -r requirements.txt.

💡 What I'd Do Next (Extensions)
Calibration — use CalibratedClassifierCV to ensure predicted probabilities are well-calibrated; critical when using probabilities directly for business decisions like CLV weighting
SHAP interaction values — compute pairwise SHAP interactions to reveal which feature combinations drive churn together (e.g. high monthly charges + month-to-month contract)
Survival analysis — replace binary churn with time-to-churn using a Cox proportional hazards model; answers "when will this customer churn?" not just "will they?"
Streamlit dashboard — deploy an interactive tool where a retention manager enters a customer's profile and gets their churn probability, top SHAP drivers, and recommended action in real time
A/B test framework — design a holdout group to measure the actual retention rate of customers who received offers vs those who didn't; validate the £Z net benefit estimate
🔗 Related Projects
House Price Regression — supervised regression, Ridge/Lasso, feature engineering
Heart Disease Classifier — binary classification, threshold tuning, AUC-ROC
Customer Segmentation — unsupervised K-Means, PCA, business personas
Credit Card Fraud Detection — extreme imbalance, SMOTE, anomaly detection
Sales Forecasting — SARIMA, XGBoost lag features, Prophet