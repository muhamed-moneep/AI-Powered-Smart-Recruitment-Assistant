# AI-Powered Smart Recruitment Assistant
- ITI CodeCamp - AI Professional Dimploma - Gradiuation Project 
A machine learning project that predicts whether a candidate who has completed job-related training is actively looking to change jobs. The goal is to help recruiters and HR teams prioritize outreach toward candidates who are genuinely in the market for a new role, instead of spending equal effort on everyone in a training/enrollment pool.

## What this project does

Given a candidate's profile (education, experience, current company, training hours completed, etc.), the model predicts:

- **`target = 1`** → candidate is likely looking for a job change
- **`target = 0`** → candidate is likely to stay put

Three classifiers are trained and compared: **Logistic Regression**, **Decision Tree**, and **Gaussian Naive Bayes**. The **Decision Tree** came out on top on this dataset (F1 = 0.626, AUC = 0.793).

## Project files

| File | Purpose |
|---|---|
| `AI_Powered_Smart_Recruitment_Assistant_.ipynb` | Main Jupyter notebook: cleaning, encoding, modeling, evaluation |
| `aug_train.csv` | Labeled training data (has `target` column) |
| `aug_test.csv` | Unlabeled data (no `target` column) — used for inference/holdout style demo |
| `REPORT.md` | Results, metrics, and business insights |
| `DOCUMENTATION.md` | Technical walkthrough of the notebook, step by step |

## Dataset

This uses the well-known **HR Analytics: Job Change of Data Scientists** style dataset. Each row is a candidate enrolled in training, described by:

- `enrollee_id`, `city`, `gender` — dropped before modeling (identifiers / high-cardinality / mostly not useful as-is)
- `city_development_index` — development index of the candidate's city
- `relevent_experience` — has relevant experience or not
- `enrolled_university` — current enrollment status
- `education_level`, `major_discipline`
- `experience` — years of experience
- `company_size`, `company_type` — current employer info
- `last_new_job` — years since last job change
- `training_hours` — hours of training completed
- `target` — 1 = looking for job change, 0 = not (only present in `aug_train.csv`)

Combined dataset size: **21,287 rows** (19,158 labeled train + 2,129 unlabeled test).

## How to run it

1. **Install dependencies**

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

2. **Keep the CSVs next to the notebook.** The notebook reads them with relative paths:

   ```python
   df_train = pd.read_csv('aug_train.csv')
   df_test  = pd.read_csv('aug_test.csv')
   ```

3. **Run the notebook top to bottom**, e.g.:

   ```bash
   jupyter notebook AI_Powered_Smart_Recruitment_Assistant_.ipynb
   ```

   Cells are organized in this order: imports → load data → explore → clean missing values → encode categoricals → train/test split → scale features → train Logistic Regression / Decision Tree / Naive Bayes → evaluate & compare → feature importance → insights.

## Results at a glance

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|---|---|---|---|---|---|
| **Decision Tree** | 0.774 | 0.532 | 0.760 | **0.626** | **0.793** |
| Naive Bayes | 0.752 | 0.502 | 0.524 | 0.512 | 0.753 |
| Logistic Regression | 0.715 | 0.453 | 0.704 | 0.551 | 0.768 |

See `REPORT.md` for the full breakdown and business takeaways.

## Known limitations / next steps

- `city` and `gender` are dropped entirely rather than encoded — potentially useful signal (especially `city`, which correlates with `city_development_index`) is lost.
- `company_type`, `enrolled_university`, and `major_discipline` are encoded with `LabelEncoder`, which imposes an artificial numeric order on categories that are not ordinal. One-hot encoding would be more correct for a linear model like Logistic Regression.
- Missing-value imputation (mode-filling, label encoding) is fit on the **combined** train+test frame before the train/test split, which is a mild form of data leakage. In a stricter pipeline, these steps should be fit on the training set only.
- Only three baseline models are compared; no hyperparameter tuning (e.g., `GridSearchCV`) was performed.
- The dataset is imbalanced (~25% positive class), which is why `class_weight='balanced'` is used — but techniques like SMOTE or threshold tuning were not explored.

## License / Data source

Dataset resembles the public "HR Analytics: Job Change of Data Scientists" dataset commonly used for educational/demo purposes. Verify licensing before any commercial use.
