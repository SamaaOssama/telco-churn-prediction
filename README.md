# 📉 ChurnGuard — Telco Customer Churn Predictor

A machine learning web application that predicts whether a telecom customer is likely to churn, built as part of an end-to-end ML project covering data cleaning, EDA, feature engineering, model comparison, and deployment.

**🔗 Live App:** [https://churnguardapp.streamlit.app](https://churnguardapp.streamlit.app)

---

## 📖 Overview

Customer churn — when a customer stops doing business with a company — is one of the most costly problems telecom providers face. This project builds a complete machine learning pipeline on the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) to predict churn risk from customer demographics, account details, and subscribed services, then deploys the best-performing model as an interactive web app.

Given a customer's profile, the app predicts:

- Whether they are **likely to churn** or **likely to stay**
- The **churn probability**, so risk can be prioritized rather than treated as a strict yes/no

## ✨ Features

- Clean, organized input form covering customer demographics, account/billing details, and subscribed services
- Real-time churn prediction using a trained **Naive Bayes** classifier
- Clear visual prediction result with churn probability displayed as both a percentage and a progress indicator
- Expandable panel showing the exact encoded feature values used for the prediction (for transparency/debugging)

## 🧠 Model

Eleven classification models were trained and compared — six individual classifiers (Logistic Regression, Decision Tree, Random Forest, KNN, SVM, Naive Bayes) and five ensemble methods (Bagging, AdaBoost, Gradient Boosting, Voting Classifier, Stacking Classifier).

**Naive Bayes** was selected as the final deployed model:

| Metric   | Score |
| -------- | ----- |
| F1-Score | 0.620 |
| Recall   | 76.7% |
| Accuracy | 75.0% |
| ROC-AUC  | 0.830 |

It was chosen for its **highest F1-score and recall** among all 11 models — in a churn-prediction context, catching as many true churners as possible (recall) matters more than avoiding false alarms, since a missed churner represents a lost customer with no retention intervention.

## 🛠️ Tech Stack

- **Python 3**
- **scikit-learn** — model training and evaluation
- **pandas** — data preprocessing
- **Streamlit** — web app framework and deployment
- **joblib** — model serialization

## 📁 Project Structure

```
telco-churn-prediction/
├── app.py                  # Streamlit application
├── churn_model.pkl         # Trained Naive Bayes model
├── model_features.pkl      # Expected feature schema (column names & order)
├── requirements.txt        # Python dependencies
└── README.md
```

## 🚀 Running Locally

1. Clone this repository:

   ```bash
   git clone https://github.com/SamaaOssama/telco-churn-prediction.git
   cd telco-churn-prediction
   ```

2. (Recommended) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate      # Mac/Linux
   venv\Scripts\activate         # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:

   ```bash
   streamlit run app.py
   ```

5. Open the URL shown in your terminal (typically `http://localhost:8501`).

## 📊 Data Pipeline Summary

The model behind this app was built through a full preprocessing and feature engineering pipeline:

- **Cleaning:** handled 11 hidden missing values in `TotalCharges`, removed redundant categorical placeholders, verified no duplicates or outliers
- **Feature Engineering:** created `IsNewCustomer` (tenure ≤ 12 months) and `NumAdditionalServices` (count of add-on subscriptions), label-encoded binary features, one-hot encoded nominal categories
- **Train/Test Split:** 80/20 stratified split (`random_state=42`) to preserve class balance
- **Scaling:** `StandardScaler` fitted on training data only, to prevent data leakage

## 🔮 Future Improvements

- Address class imbalance via `class_weight='balanced'`, SMOTE, or threshold tuning
- Hyperparameter tuning (`GridSearchCV` / `RandomizedSearchCV`)
- Model explainability via SHAP values
- Adjustable classification threshold based on business cost trade-offs
