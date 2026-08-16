# 🛍️ Customer Segmentation — K-Means Clustering

> Unsupervised machine learning to segment mall customers by income and spending behaviour.  
> **K-Means++ · k=5 · Silhouette=0.55 · PCA 2D visualisation · 5 actionable business personas**

---

## 📌 Project Overview

Applied K-Means clustering to segment 200 mall customers into distinct behavioural groups using Annual Income and Spending Score as the primary feature space. The project covers the full unsupervised learning workflow: exploratory analysis, feature scaling, optimal k selection via elbow method and silhouette scores, PCA dimensionality reduction for visualisation, and translation of clusters into actionable marketing personas.

**This is the third project in a supervised → unsupervised ML portfolio progression:**

| Project | Type | Level |
|---|---|---|
| [House Price Regression](https://github.com/YOUR_USERNAME/house-price-regression) | Supervised — Regression | Beginner |
| [Heart Disease Classifier](https://github.com/YOUR_USERNAME/heart-disease-classifier) | Supervised — Classification | Beginner |
| **Customer Segmentation** ← you are here | Unsupervised — Clustering | Beginner |

**Key skills demonstrated:**
- Unsupervised learning — no labels, no ground truth
- Feature scaling with `StandardScaler` (why it matters for distance-based algorithms)
- Elbow method + silhouette score double-validation for optimal k
- PCA for cluster visualisation and explained variance
- Translating cluster centroids into named business personas
- Marketing recommendations from data-driven segmentation

> **Key conceptual contrast with supervised learning:** There is no target variable to predict and no accuracy metric to optimise. The elbow plot and silhouette score are heuristics that guide judgment — not definitive answers. Explaining this distinction is what demonstrates real understanding of unsupervised learning.

---

## 📊 Results Summary

| Metric | Value |
|---|---|
| Optimal clusters (k) | 5 |
| Silhouette score | 0.55 |
| PCA variance retained (2D) | 79.3% |
| Algorithm | K-Means++ |
| Feature space | Annual Income + Spending Score (scaled) |

> ⚠️ Replace with your actual silhouette score and PCA variance from the Phase 3 & 4 console output.

### The 5 Customer Personas

| Cluster | Persona | Avg Income | Avg Spending | Size | Priority |
|---|---|---|---|---|---|
| 0 | 💎 High Value | High | High | ~45 | HIGH |
| 1 | 💰 Careful Spenders | High | Low | ~35 | HIGH |
| 2 | 🛍️ Impulsive Buyers | Low | High | ~45 | MEDIUM |
| 3 | 📦 Budget Conscious | Low | Low | ~40 | LOW |
| 4 | ⚖️ Mid-Tier | Medium | Medium | ~35 | MEDIUM |

> ⚠️ Replace cluster indices, sizes, and income/spending values with your actual Phase 2 cluster profile output.

---

## 🗂️ Project Structure

```
customer-segmentation/
│
├── data/
│   ├── Mall_Customers.csv          # Raw dataset (from Kaggle)
│   └── customers_clustered.csv    # Output of Phase 2 — with cluster labels
│
├── outputs/
│   ├── phase1_gender_split.png
│   ├── phase1_distributions.png
│   ├── phase1_distributions_by_gender.png
│   ├── phase1_pairplot.png
│   ├── phase1_correlation_heatmap.png
│   ├── phase2_elbow_silhouette.png
│   ├── phase2_cluster_scatter.png
│   ├── phase2_silhouette_plot.png
│   ├── phase3_cluster_visualisation.png
│   ├── phase4_cluster_bar_charts.png
│   ├── phase4_radar_chart.png
│   └── phase4_recommendations_table.png
│
├── seg_phase1_eda.py               # EDA, distributions, pairplot
├── seg_phase2_kmeans.py            # K-Means, elbow method, silhouette
├── seg_phase3_4_personas.py        # PCA, persona naming, radar chart, recommendations
├── requirements.txt
└── README.md
```

---

## 🔍 Key Findings

### 1. Income vs Spending Score reveals natural clusters
Even before any algorithm runs, the pairplot shows 5 visually distinct groupings in the Income vs Spending Score space. This is the foreshadowing that makes the K-Means result feel intuitive rather than arbitrary.

### 2. Scaling is non-negotiable for K-Means
K-Means uses Euclidean distance. Without `StandardScaler`, annual income (range $15k–$137k) would dominate spending score (1–100) simply due to its larger numeric range — creating clusters driven by income magnitude, not actual behaviour.

### 3. Elbow + silhouette agree on k=5
Two independent validation methods both point to k=5:

| k | Inertia | Silhouette |
|---|---|---|
| 2 | — | 0.38 |
| 3 | — | 0.44 |
| 4 | — | 0.50 |
| **5** | **—** | **0.55 ← peak** |
| 6 | — | 0.51 |

> ⚠️ Fill in inertia values from your Phase 2 console output.

### 4. PCA retains ~79% of variance in 2D
Reducing three features (Age, Income, Spending) to two principal components retains the majority of the information — enough for a meaningful cluster visualisation. PC1 primarily captures the Income-Spending axis; PC2 captures age variation.

### 5. Gender is not a cluster driver
Despite a 56/44 female/male split, gender is evenly distributed across most clusters. Income and spending behaviour — not demographics — drive the segmentation. This is a meaningful finding: a gender-based marketing strategy would miss the real structure in this data.

### 6. Actionable marketing recommendations

| Persona | Strategy | Channel | Watch out |
|---|---|---|---|
| 💎 High Value | Loyalty programmes, VIP access, early launches | Personalised email, concierge | Churn risk — prioritise retention |
| 💰 Careful Spenders | ROI-focused messaging, quality guarantees | Detailed email, comparison guides | Under-monetised — high potential |
| 🛍️ Impulsive Buyers | Flash sales, bundles, time-limited offers | Push notifications, social ads | Discount dependency |
| 📦 Budget Conscious | Entry-level products, payment plans | SMS, low-cost digital | Low margin — focus on volume |
| ⚖️ Mid-Tier | Cross-selling, upgrade nudges | Newsletters, in-app | Generic messaging risks being ignored |

---

## 📈 Visual Highlights

### Pairplot — Cluster Structure Visible Before Modelling
![Pairplot](outputs/phase1_pairplot.png)

### Elbow Method + Silhouette Scores
![Elbow and Silhouette](outputs/phase2_elbow_silhouette.png)

### Cluster Scatter (Original Space)
![Cluster Scatter](outputs/phase2_cluster_scatter.png)

### Silhouette Plot — Per Sample
![Silhouette Plot](outputs/phase2_silhouette_plot.png)

### PCA Projection + Original Space Side by Side
![PCA Visualisation](outputs/phase3_cluster_visualisation.png)

### Feature Profiles per Cluster
![Bar Charts](outputs/phase4_cluster_bar_charts.png)

### Persona Radar Chart
![Radar Chart](outputs/phase4_radar_chart.png)

### Marketing Recommendations Table
![Recommendations](outputs/phase4_recommendations_table.png)

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/customer-segmentation.git
cd customer-segmentation
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get the data
Download `Mall_Customers.csv` from [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) and place it in the `data/` folder.

### 4. Run phases in order
```bash
python seg_phase1_eda.py
python seg_phase2_kmeans.py
python seg_phase3_4_personas.py
```

Each script saves outputs to the `outputs/` folder and prints a summary to the console.

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

- **Hierarchical clustering** — compare K-Means results with Agglomerative Clustering and a dendrogram; does the hierarchy agree with the 5 personas?
- **DBSCAN** — density-based clustering that can detect outlier customers as noise; useful for identifying anomalous spending patterns
- **Include Age as a third clustering dimension** — rerun K-Means on all three features and compare personas; does adding age split any cluster further?
- **Streamlit dashboard** — interactive tool where a marketing manager inputs a customer's age, income, and spending score and gets their persona and recommended strategy in real time
- **RFM analysis** — extend to a transactional dataset (Recency, Frequency, Monetary) for a more business-realistic segmentation

---

## 🔗 Related Projects

- [House Price Regression](https://github.com/YOUR_USERNAME/house-price-regression) — supervised regression on the Ames Housing dataset
- [Heart Disease Risk Classifier](https://github.com/YOUR_USERNAME/heart-disease-classifier) — binary classification with threshold tuning on the Cleveland dataset

---

## 👤 Author

**Your Name**  
[LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername) · [Kaggle](https://kaggle.com/yourusername)

---

## 📄 License

MIT License — free to use, adapt, and share with attribution.
