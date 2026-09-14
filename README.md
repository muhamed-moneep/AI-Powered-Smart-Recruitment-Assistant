# 💼 Smart Recruitment Assistant
### AI Professional Diploma — Graduation Project

An intelligent recruitment screening system that predicts whether a candidate should advance in the hiring pipeline, built on a large-scale synthetic recruitment dataset and benchmarked across five classic Machine Learning models.

---

## 📌 Project Overview

Modern organizations receive hundreds (or thousands) of applications per vacancy, making manual screening slow and inconsistent. This project builds a data-driven hiring-decision classifier that:

- Cleans and prepares raw candidate data for modeling
- Trains and compares **five Machine Learning models**
- Evaluates each model with standard classification metrics
- Surfaces **business insights** on what actually drives hiring decisions
- Produces a **Top-10 candidate ranking system** to give recruiters an actionable shortlist

---

## 📊 Dataset

**Resume Screening Dataset (200K Candidates)**
🔗 [https://www.kaggle.com/datasets/rhythmghai/resume-screening-dataset-200k-candidates](https://www.kaggle.com/datasets/rhythmghai/resume-screening-dataset-200k-candidates)

- **Size:** ~200,000 candidate records
- **Type:** Synthetic, tabular recruitment/hiring data
- **Task:** Binary classification — predict the hiring decision / candidate advancement outcome
- **License:** As specified on the Kaggle dataset page — verify before redistribution

> ⚠️ Dataset files are not included in this repository due to size and licensing. Download directly from the Kaggle link above and place the raw file(s) in `data/raw/`.

---

## 🎯 Objectives

1. Prepare a large, messy recruitment dataset for reliable modeling
2. Build and compare multiple classifiers on the same candidate-screening task
3. Identify the most influential hiring factors
4. Translate model output into recruiter-facing insights and a ranked shortlist

---

## 🛠️ Project Structure

```
Smart-Recruitment-Assistant/
│
├── data/
│   ├── raw/                     # Original dataset (downloaded from Kaggle)
│   └── processed/                # Cleaned & feature-engineered dataset
│
├── notebooks/
│   └── Smart_Recruitment_Assistant.ipynb   # Full end-to-end notebook
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   └── naive_bayes.pkl
│
├── reports/
│   ├── model_performance_comparison.csv
│   └── candidate_ranking_top10.csv
│
├── dashboard/
│   └── recruitment_dashboard.html
│
└── README.md
```

---

## 🧹 1. Data Preparation

**Missing Value Handling**
- Categorical fields: impute with an explicit `"Unknown"` category (missingness itself can be informative in HR data) or the column mode where appropriate
- Numerical fields: median/mean imputation depending on distribution and skew
- Rows/columns with excessive missingness assessed for removal vs. imputation

**Feature Engineering**
- Bucketing continuous fields (e.g., experience, age, scores) into meaningful tiers
- Deriving new signals (e.g., relevant-experience flags, seniority level, profile-completeness score)
- Encoding categorical variables (Label Encoding / One-Hot Encoding as appropriate per model)

**Data Transformation**
- Feature scaling (StandardScaler / MinMaxScaler) for scale-sensitive models (Logistic Regression, KNN)
- Train/test split with stratification on the target to preserve class balance
- Class imbalance handling if needed (`class_weight='balanced'`, resampling, etc.)

---

## 🤖 2. Machine Learning Models

Five classifiers are trained and compared on the same prepared dataset:

| Model | Type | Notes |
|---|---|---|
| **Logistic Regression** | Linear baseline | Fast, interpretable coefficients — good for explaining hiring factors to non-technical stakeholders |
| **Random Forest** | Ensemble of trees | Captures non-linear feature interactions; provides feature importance |
| **Decision Tree** | Single tree | Fully interpretable, visualizable splits; used as a baseline before ensembling |
| **K-Nearest Neighbors (KNN)** | Instance-based | Simple, non-parametric; sensitive to feature scaling and dataset size |
| **Naive Bayes** | Probabilistic | Extremely fast baseline; assumes feature independence |

Each model is trained on an identical train/test split for a fair, apples-to-apples comparison.

---

## 📈 3. Model Evaluation

Every model is evaluated on the held-out test set using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

A consolidated **model comparison table and chart** rank all five models side by side across every metric, making it clear which model(s) best balance precision (avoiding false positives on unqualified candidates) against recall (not missing strong candidates).

---

## 💡 4. Business Insights

Beyond raw metrics, the project translates model output into recruiter-facing insights:

- **Most Important Hiring Factors** — feature importance (from Random Forest / Decision Tree) highlighting which candidate attributes most influence the hiring decision
- **Candidate Profile Trends** — how hiring outcomes vary across education level, experience, skills, and other candidate segments
- **Recruitment Statistics** — overall pass-rate, class balance, and distribution of key candidate attributes across the applicant pool

---

## 🏆 5. Candidate Ranking System (Top 10)

Using the best-performing model's predicted probabilities, the system generates a ranked shortlist of the **Top 10 candidates** most likely to be strong hires — giving recruiters an immediately actionable list instead of manually reviewing the full applicant pool.

---

## 🧰 Technologies Used

- **Language:** Python
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** Scikit-Learn
- **Environment:** Jupyter Notebook

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone <repo-url>
cd Smart-Recruitment-Assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset from Kaggle and place it in data/raw/

# 4. Launch the notebook
jupyter notebook notebooks/Smart_Recruitment_Assistant.ipynb
```

---

## 📄 Deliverables

- [ ] Jupyter Notebook (`.ipynb`) — full pipeline, end to end
- [ ] Trained models (`.pkl`)
- [ ] Model performance comparison report
- [ ] Interactive recruitment dashboard
- [ ] Top-10 candidate ranking output
- [ ] Project report (2–4 pages)
- [ ] Presentation slides (5–10 slides)

---

## 👤 Author

AI Professional Diploma — Graduation Project
*Smart Recruitment Assistant*
