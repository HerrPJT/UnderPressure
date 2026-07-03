# Under Pressure

A Machine Learning project analyzing how daily digital habits — screen time, sleep, notifications, caffeine intake — relate to stress levels, using a sample of 15,000 individuals.

> Project for the *Machine Learning* course, Licenciatura em Engenharia Eletrotécnica e de Computadores, Faculdade de Ciências e Tecnologia, Universidade de Coimbra.

**Authors**
- Pedro José Carvalho Tomás — Nº2023211605 — uc2023211605@student.uc.pt
- Samuel Baptista Figueira — Nº2023211662 — uc2023211662@student.uc.pt

---

## Overview

Over the last 20 years digital technologies have become nearly ubiquitous, but so have burnout, stress, and mental fatigue. This project uses supervised classification to investigate how factors such as daily screen time and sleep habits relate to stress levels, and compares which algorithms perform best under different classification granularities.

The original target variable, `stress_level`, is a continuous score from 0–10. It is discretized into three separate classification scenarios:

| Scenario | Classes | Description |
|---|---|---|
| **A** | 2 | Binary: No Stress (0) vs. Stress (1) |
| **B** | 5 | Stress level grouped into 5 bins (0–4) |
| **C** | 10 | Stress level grouped into 10 bins (0–9) |

## Dataset

- **Source:** [Sleep, Screen Time and Stress Analysis](https://www.kaggle.com/datasets/jayjoshi37/sleep-screen-time-and-stress-analysis) (Kaggle, by Jay Joshi)
- **Size:** 15,000 rows, 13 columns
- **No missing or duplicate values**

Columns include: `user_id`, `age`, `gender`, `occupation`, `daily_screen_time_hours`, `phone_usage_before_sleep_minutes`, `sleep_duration_hours`, `sleep_quality_score`, `caffeine_intake_cups`, `physical_activity_minutes`, `notifications_received_per_day`, `mental_fatigue_score`, and the target `stress_level`.

## Pipeline

1. **Data loading & inspection** — check for missing/duplicate values, rename columns for readability.
2. **Exploratory Data Analysis (EDA)** — descriptive statistics, correlation matrix, scatter plots, box plots, class distribution per scenario.
3. **Preprocessing**
   - Dropped `user_id` (identifier) and `mental_fatigue_score` (co-target / potential data leakage).
   - One-Hot Encoding for categorical variables (`occupation`, `gender`).
   - Target discretization into the 3 scenarios (`numpy.where`, `numpy.select`, `pandas.cut`).
   - Train/test split (75/25), stratified, `random_state=42`.
   - Feature scaling with `StandardScaler`.
   - Class balancing with **SMOTE** on the training set (all scenarios show class imbalance, especially scenario C).
4. **Modeling** — five models, each hyperparameter-tuned with **Optuna** (cross-validation as the optimization metric):
   - **Neural Network** — `MLPClassifier` (scenario A) / PyTorch MLP (scenarios B and C)
   - **Support Vector Machine (SVM)**
   - **Random Forest Classifier**
   - **Naive Bayes** (no tuning — used as a fast baseline)
   - **Voting Classifier** (ensemble of the above, excluding PyTorch models in the multiclass scenarios)

   Every model is trained both on the original data and on SMOTE-resampled data.
5. **Evaluation** — classification reports, normalized confusion matrices, ROC curves, Mean Absolute Error (MAE), tolerance-based accuracy (±1 / ±2 classes), feature importance (Random Forest), and learning curves.

## Key Findings

- **Scenario A (binary):** all models perform very well (~92% accuracy / F1), with SVM showing the best ROC-AUC — the classes are close to linearly separable.
- **Scenario B (5 classes):** performance drops noticeably. Random Forest and the Voting Classifier perform best; SMOTE improves minority-class recall and stabilizes predictions. Mean error stays under 1 class (MAE ≈ 0.3).
- **Scenario C (10 classes):** performance drops further, but with a tolerance of ±2 classes, accuracy climbs back to nearly 100%. Voting Classifier scores highest, though Random Forest is preferred as a lighter-weight alternative.
- **Most influential features (Random Forest):** `daily_screen_time` and `sleep_quality_score` consistently dominate across all three scenarios, confirming the correlations found in EDA (screen time: +0.88, sleep quality: −0.86 with stress).
- **SMOTE trade-off:** slightly lowers overall accuracy but improves recall for minority (low-stress) classes and makes F1-scores more consistent across classes — most beneficial for SVM.
- **Overfitting:** Random Forest shows overfitting in the multiclass scenarios (learning curves), flagged as an area for future tuning.

## Repository Contents

| File | Description |
|---|---|
| `UnderPressurePythonScript.py` | Full pipeline: EDA, preprocessing, model training/tuning, evaluation |
| `Relatório_Projeto_PedroTomas_SamuelFigueira.pdf` | Full written report (in Portuguese), including all figures and discussion |

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
optuna
torch
imbalanced-learn
```

Install with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn optuna torch imbalanced-learn
```

## Usage

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/jayjoshi37/sleep-screen-time-and-stress-analysis) and place it at `archive/sleep_mobile_stress_dataset_15000.csv`.
2. Run the script:

```bash
python UnderPressurePythonScript.py
```

The script is structured in notebook-style cells (`# %%`) and can also be opened directly in Jupyter or VS Code.

## Future Work

- More advanced class-balancing techniques
- Additional models such as XGBoost or deeper neural architectures
- Incorporating factors not present in the dataset (e.g., genetics, traumatic events) that may also influence stress

## Disclaimer

Generative AI tools (Claude and Gemini) were used as auxiliary aids for research and for customizing comparison plots and assisting with target discretization. All modeling, implementation, and interpretation of results were carried out by the authors.

## References

1. Jay Joshi. *Sleep, Screen Time and Stress Analysis*. Kaggle. https://www.kaggle.com/datasets/jayjoshi37/sleep-screen-time-and-stress-analysis
2. PyTorch documentation. https://docs.pytorch.org/docs/2.12/index.html
3. Optuna documentation. https://optuna.readthedocs.io/en/stable/
