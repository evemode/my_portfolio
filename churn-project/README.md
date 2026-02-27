# 📉 Customer Churn Prediction

## TL;DR
- Binary classification project for customer churn prediction
- Business-focused approach with recall-oriented decision making
- Compared Logistic Regression, Random Forest, Gradient Boosting, and MLP
- Random Forest selected as final model (threshold = 0.3)
- Deployed as an interactive Dash web application

---

## 📌 Project Overview
This project focuses on predicting customer churn using supervised machine learning on a structured telecom dataset.  
The goal is to identify customers who are likely to leave the service and support **business-oriented retention decisions**.

The project follows a full **end-to-end data science workflow**, from raw data analysis to a deployment-ready web application.

---

## 🎯 Business Objective
Customer churn prediction is framed as a **binary classification problem**, where:

- **1** → customer is likely to churn  
- **0** → customer is likely to stay  

The main business priority is **high recall**, ensuring that as many churned customers as possible are correctly identified, even at the cost of lower precision.

---

## 📂 Project Structure
```text
churn-project/
  app/
    dash_app.py
  data/
    raw/
    processed/
  models/
    churn_rf_bundle.pkl
  notebooks/
    01_eda.ipynb
    02_preprocessing.ipynb
    03_modeling.ipynb
  reports/
    correlation_matrix_numeric.png
  pyproject.toml
  poetry.lock
  README.md
```

---

## 🔍 Exploratory Data Analysis
EDA includes:
- Analysis of churn class imbalance
- Relationship between churn and contract type
- Impact of additional services (Internet, Security, Streaming)
- Numerical feature distributions
- Correlation analysis

Key insights from EDA guided preprocessing and model selection decisions.

---

## 🧹 Data Preprocessing
- Missing values handled
- Categorical variables encoded using **One-Hot Encoding**
- Numerical variables preserved
- Final dataset contains boolean, integer, and float features

Processed data is stored in `data/processed/processed.csv`.

---

## 🤖 Models Evaluated
The following models were trained and compared:

- Logistic Regression (baseline, interpretable)
- Random Forest
- Gradient Boosting
- Multilayer Perceptron (Neural Network)

Evaluation metrics:
- Precision
- Recall
- F1-score
- ROC-AUC

---

## 📊 Model Comparison (Threshold = 0.3)

| Model | Precision | Recall | F1 | ROC-AUC |
|------|-----------|--------|----|---------|
| Logistic Regression | 0.517 | 0.749 | 0.611 | 0.841 |
| **Random Forest** | **0.522** | **0.762** | **0.620** | 0.840 |
| Gradient Boosting | 0.517 | 0.749 | 0.611 | 0.841 |
| MLP (Neural Network) | 0.506 | 0.741 | 0.602 | 0.822 |

The comparison shows that tree-based ensemble models outperform neural networks on this structured dataset, with Random Forest achieving the best recall–F1 balance.

---

## 🔎 Model Explainability (SHAP)

To better understand model decisions, SHAP (SHapley Additive exPlanations) 
was applied to the final Random Forest model.

SHAP provides feature-level contributions to individual predictions 
and helps identify global churn drivers.

### Key findings from SHAP analysis:

- **Tenure** is the strongest predictor of churn.  
  Customers with shorter tenure have significantly higher churn probability.

- **Contract type** plays a critical role.  
  Two-year contracts strongly reduce churn risk.

- **Fiber optic internet service** is associated with higher churn probability.

- Pricing-related features such as **TotalCharges** and **MonthlyCharges**
  also contribute to churn behavior.

The explainability analysis confirms business intuition and 
provides actionable insights for retention strategy.

---

## ✅ Final Model Selection
**Random Forest** was selected as the final model because it:
- Achieved the best balance between **Recall** and **F1-score**
- Performed consistently across cross-validation
- Effectively captured non-linear relationships
- Is well-suited for structured tabular data

The decision threshold was adjusted to **0.3** to prioritize recall, aligning with the business objective of churn prevention.

---

## 🚀 Web Application (Dash)
The final model is deployed via a **Dash web application**.

### Features:
- Interactive input form for customer features
- Adjustable churn decision threshold
- Output of churn probability and predicted label
- Visualization of probability vs threshold

The application loads a **serialized sklearn pipeline using joblib**, ensuring consistent preprocessing and inference.

---

## ▶️ How to Run the App

### Requirements
- Python 3.10+

### 1️⃣ Install dependencies
```bash
poetry install
poetry run python app/dash_app.py
http://127.0.0.1:8050
```

🛠 Tech Stack

Python
Pandas, NumPy
Scikit-learn
Dash, Plotly
Joblib
Poetry

🧠 Key Takeaways

Tree-based ensemble models outperform neural networks on structured tabular data
Threshold tuning is critical for business-oriented metrics such as recall
Deployment-ready pipelines are essential for real-world ML applications
Neural networks are not always the best choice for tabular datasets

📌 Author

Oleksandr Sielikhov
Junior Data Scientist
Portfolio project focused on applied machine learning and deployment