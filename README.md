<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Decision%20Tree-Best%20Model-228B22?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Precision-100%25-brightgreen?style=for-the-badge"/>
<img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Domain-Cardiology%20AI-red?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Deployment-Pickle-lightgrey?style=for-the-badge"/>

<br/><br/>

# ❤️ ECG-Based Heart Disease Classification System

### *Cardiovascular risk prediction using ECG patterns and clinical biomarkers — Cardiology Informatics Lab*

<br/>

> **Clinical Mission:** Every missed heart disease diagnosis costs a life.  
> This project builds a statistically validated, clinically interpretable ML classification system to identify patients at cardiovascular risk — using ECG readings, lifestyle factors, and laboratory values from the Erbil Heart Disease Dataset.

<br/>

---

</div>

## 📌 Table of Contents

- [Project Background](#-project-background)
- [Dataset & Clinical Dictionary](#-dataset--clinical-dictionary)
- [Project Workflow](#-project-workflow)
- [Statistical Validation](#-statistical-validation)
- [Feature Engineering](#-feature-engineering)
- [Models & Results](#-models--results)
- [Model Selection Rationale](#-model-selection-rationale)
- [Feature Importance](#-feature-importance)
- [Deployment](#-deployment)
- [Tech Stack](#-tech-stack)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)

---

## 🏥 Project Background

**Role:** Senior Data Scientist  
**Department:** Cardiology Informatics Lab

This project goes far beyond a standard classification task. It applies **full statistical hypothesis testing** to validate every feature's clinical relevance before modeling — a level of rigor expected in medical AI, not just in Kaggle competitions.

The system classifies patients as cardiovascular disease positive or negative using ECG waveform patterns, blood pressure readings, cholesterol levels, lifestyle data, and comorbidities — and delivers feature importance rankings that cardiologists can act on.

---

## 📂 Dataset & Clinical Dictionary

**Source:** Erbil Heart Disease Dataset  
**File:** `ECG-Dataset.csv`  
**Target:** `target` — Binary (0 = No disease, 1 = Disease present)

| Feature | Type | Clinical Meaning |
|---|---|---|
| `age` | Numeric | Primary cardiovascular risk factor — risk increases with age |
| `sex` | Binary | Male patients show higher early CVD risk |
| `smoke` | Binary | Major modifiable cardiovascular risk factor |
| `years` | Numeric | Cumulative smoking exposure (pack-years proxy) |
| `fh` | Binary | Family history — inherited/genetic cardiac risk |
| `ldl` | Numeric | Elevated LDL strongly linked to atherosclerosis |
| `chp` | Categorical (1–4) | Cholesterol phenotype class |
| `height` / `weight` | Numeric | Used for BMI computation |
| `bpsys` / `bpdias` | Numeric | Systolic & diastolic blood pressure |
| `hr` | Numeric | Abnormal HR can indicate conduction disorders |
| `ihd` | Binary | Existing ischemic heart disease |
| `dm` | Binary | Diabetes — major cardiovascular risk multiplier |
| `htn` | Binary | Hypertension status |
| `active` | Binary | Physical inactivity increases metabolic & cardiac risk |
| `ivsd` | Numeric | Interventricular septal thickness — LVH indicator |
| `ecgpatt` | Categorical (1–4) | ECG waveform type — ischemia/conduction patterns |
| `qwave` | Binary | Pathological Q-wave — prior myocardial infarction marker |
| **`target`** | Binary | 🎯 **Disease outcome — the prediction target** |

---

## 🔬 Project Workflow

```
Raw Clinical Dataset
   │
   ├── Phase 1: Data Understanding & Clinical Context
   │       ├── Row/column audit, dtype verification
   │       ├── Missing values & duplicate checks
   │       ├── Categorical unique value analysis
   │       └── Countplots per categorical variable
   │
   ├── Phase 2: In-Depth EDA
   │       ├── Univariate: Mean, std, skewness, kurtosis, KDE, histograms, boxplots
   │       ├── Bivariate: KDE split by target class per numeric feature
   │       ├── ECG pattern × disease crosstab + bar plots
   │       └── Pearson & Spearman correlation heatmaps
   │
   ├── Phase 3: Statistical Hypothesis Testing ← Medical-grade rigor
   │       ├── Point-biserial correlation: Numeric vs Target
   │       ├── Chi-square test: Categorical vs Target
   │       ├── Spearman correlation: Continuous vs Continuous
   │       └── One-Way ANOVA: Continuous vs Categorical (>2 groups)
   │
   ├── Phase 4: Feature Engineering
   │       └── 17 clinically derived features added
   │
   ├── Phase 5: Data Preparation
   │       ├── Stratified 80/20 train-test split
   │       └── Class ratio verified in both splits
   │
   ├── Phase 6: Model Development
   │       └── 5 classifiers compared
   │
   ├── Phase 7: Model Evaluation & Selection
   │       ├── Precision selected as key medical metric
   │       ├── ROC-AUC curve plotted
   │       └── Confusion matrix analysis
   │
   ├── Phase 8: Feature Importance & Explainability
   │       └── Top 7 features for lean final model
   │
   └── Phase 9: Deployment
           ├── Model saved with Pickle
           └── Load & predict pipeline demonstrated
```

---

## 📐 Statistical Validation

This project applies **formal statistical tests** to validate each feature's relevance — not just correlation heatmaps.

### Point-Biserial Correlation — Numeric Features vs Target

| Feature | Correlation | P-Value | Significant? |
|---|---|---|---|
| **age** | **0.2865** | **1.03 × 10⁻⁷** | ✅ Yes |
| years | 0.0585 | 0.2868 | ❌ No |
| height | -0.0897 | 0.1024 | ❌ No |
| weight | -0.0848 | 0.1227 | ❌ No |
| hr | -0.0095 | 0.8624 | ❌ No |
| ldl | -0.0550 | 0.3167 | ❌ No |
| bpsys | 0.0518 | 0.3464 | ❌ No |
| bpdias | -0.0324 | 0.5561 | ❌ No |

> **Key finding:** Only `age` shows statistically significant linear association with the target among raw numeric features — confirming that raw vitals alone are insufficient and domain-engineered features are essential.

### Chi-Square Test — Categorical Features vs Target

| Feature | P-Value | Significant? |
|---|---|---|
| **ecgpatt** | **1.57 × 10⁻⁶⁰** | ✅ Yes — Strongest signal |
| **qwave** | **3.53 × 10⁻¹²** | ✅ Yes |
| **ihd** | 0.000158 | ✅ Yes |
| **active** | 0.000342 | ✅ Yes |
| **lifestyle** | 0.00676 | ✅ Yes |
| **dm** | 0.04994 | ✅ Yes (borderline) |
| sex | 0.8949 | ❌ No |
| smoke | 0.4758 | ❌ No |
| htn | 0.7221 | ❌ No |

> **Standout result:** ECG pattern (`ecgpatt`) has a p-value of 10⁻⁶⁰ — the single most statistically powerful predictor. Q-waves (prior MI markers) are second.

---

## 🧪 Feature Engineering

17 clinically grounded features were engineered to capture interactions and composite risks:

| Engineered Feature | Logic | Clinical Rationale |
|---|---|---|
| `BMI` | `weight / (height/100)²` | Obesity is a cardiovascular risk factor |
| `pulse_pressure` | `bpsys - bpdias` | Elevated PP linked to arterial stiffness |
| `MAP` | `(bpsys + 2×bpdias) / 3` | Mean arterial pressure — perfusion indicator |
| `smoke_exposure` | `smoke × years` | Pack-years proxy for cumulative lung/cardiac damage |
| `ldl_age_ratio` | `ldl / age` | LDL burden relative to age |
| `ldl_smoke_interaction` | `ldl × smoke` | Dual-risk lipid + smoking synergy |
| `obese` | `BMI ≥ 30` | Binary obesity flag |
| `bp_category` | Staged 0–3 by guidelines | Hypertension staging |
| `hr_risk` | `hr > 90` | Elevated resting HR risk flag |
| `dm_ldl_interaction` | `dm × ldl` | Diabetes amplifying LDL risk |
| `ecg_major_abnormal` | ECG pattern in {3, 4} | Major waveform abnormality flag |
| `ecg_mild_abnormal` | ECG pattern in {1, 2} | Mild waveform abnormality flag |
| `atherogenic_index` | `log((ldl+1)/(chp+1))` | Atherosclerosis risk index |
| `metabolic_syndrome_score` | Sum of 4 metabolic risk flags | Combined metabolic burden |
| `smoke_bpsys_interaction` | `smoke × bpsys` | Smoking exacerbating hypertension |
| `active_lifestyle_interaction` | `active × lifestyle` | Combined activity-lifestyle protective score |
| `ivsd_risk` | `ivsd > 0` | Septal thickness risk binary |

---

## 📈 Models & Results

Five classifiers evaluated on a stratified 80/20 split (class ratios preserved):

| Model | Accuracy | Precision | Recall | F1-Score | MCC |
|---|---|---|---|---|---|
| **Decision Tree** ✅ | **0.9701** | **1.0000** | **0.9167** | **0.9565** | **0.9359** |
| Random Forest | 0.9552 | 0.9200 | 0.9583 | 0.9388 | 0.9040 |
| XGBoost | 0.9552 | 0.9200 | 0.9583 | 0.9388 | 0.9040 |
| Logistic Regression | 0.9254 | 0.9130 | 0.8750 | 0.8936 | 0.8366 |
| K-Nearest Neighbors | 0.5522 | 0.3333 | 0.2500 | 0.2857 | -0.0314 |

---

## 🏆 Model Selection Rationale

> **In a clinical setting, a False Positive (predicting disease when none exists) leads to unnecessary treatment — but a False Negative (missing a real diagnosis) can cost a patient their life.**

The selection criterion here is **Precision**: minimizing False Positives to ensure that every positive prediction is reliable.

**Decision Tree achieved Precision = 1.000** — zero false positives — making it the optimal model for clinical deployment. The lean top-7 feature model maintained this performance, confirming that feature selection improved efficiency without sacrificing diagnostic reliability.

---

## 🔍 Feature Importance

Top 7 features identified by Decision Tree importance scores (used in the final deployment model):

| Rank | Feature | Type | Clinical Meaning |
|---|---|---|---|
| 1 | `ecgpatt` | ECG | Waveform pattern — strongest predictor by far |
| 2 | `qwave` | ECG | Prior MI marker |
| 3 | `BMI` | Engineered | Obesity indicator |
| 4 | `MAP` | Engineered | Mean arterial pressure |
| 5 | `pulse_pressure` | Engineered | Arterial stiffness proxy |
| 6 | `age` | Clinical | Primary demographic risk factor |
| 7 | `metabolic_syndrome_score` | Engineered | Composite metabolic burden |

> The dominance of ECG-derived features (`ecgpatt`, `qwave`) validates the dataset's name — and confirms that waveform analysis is the clinical gold standard for cardiovascular risk stratification.

---

## 🚀 Deployment

The final Decision Tree model (top-7 features) is serialized and ready for clinical integration:

```python
import pickle

# Load the model
with open('model_final.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict on a new patient
# Features: ecgpatt, qwave, BMI, MAP, pulse_pressure, age, metabolic_syndrome_score
new_patient = [[3, 0, 0.0, 32.96, 188, 63.33, 50]]

prediction = model.predict(new_patient)
# Output: [1] → Cardiovascular Disease Detected
# Output: [0] → No Disease Detected
```

---

## 🛠 Tech Stack

```
Language      Python 3.10+
ML Library    scikit-learn, XGBoost
Statistics    scipy (pointbiserialr, chi2_contingency)
Data          pandas, NumPy
Viz           matplotlib, seaborn
Deployment    Pickle
Environment   Google Colab / Jupyter Notebook
```

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Pfuglo1/ecg-heart-disease-classification.git
cd ecg-heart-disease-classification

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy

# 3. Launch the notebook
jupyter notebook Project_5_ECG_Based_Heart_Disease_Classification_System.ipynb
```

> 📂 Place `ECG-Dataset.csv` in the same directory before running.

---

## 🗂 Project Structure

```
ecg-heart-disease-classification/
│
├── Project_5_ECG_Based_Heart_Disease_Classification_System.ipynb   # Full pipeline
├── Project_Roadmap_ECG-Based_Heart_Disease_Classification.pdf      # Project brief
├── model_final.pkl                                                  # Saved model (Pickle)
├── clean_data.csv                                                   # Engineered dataset
├── README.md                                                        # You are here
└── ECG-Dataset.csv                                                  # Raw dataset
```

---

<div align="center">

**Built at Cardiology Informatics Lab 🏥 | ECG-Driven Cardiovascular Risk AI**

*Precision = 1.000 — because in cardiology, there is no room for false confidence.*

*Found this useful? Give it a ⭐!*

</div>
