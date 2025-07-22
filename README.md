# 🛡️ Fraud Detection System – Adey Innovations Inc.

## 📌 Project Overview

This project aims to develop machine learning models for detecting fraudulent transactions in both **e-commerce** and **banking** systems. It combines behavioral features, time-based patterns, and IP geolocation to enhance detection accuracy.

---

## 📂 Datasets

| File                     | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `Fraud_Data.csv`         | E-commerce transactions with fraud labels and features like time, device, etc. |
| `IpAddress_to_Country.csv` | Maps numeric IP address ranges to countries                                 |
| `creditcard.csv` (zipped) | Bank transactions with anonymized variables (V1 to V28), amount, and class  |

---

## ✅ Tasks Completed

### 1. **Data Loading & Cleaning**
- Loaded datasets using pandas
- Handled missing values and duplicates
- Converted timestamps to datetime format

### 2. **Exploratory Data Analysis (EDA)**
- Visualized fraud rates by browser, hour, and country
- Checked class imbalance (severe in bank data, moderate in e-commerce)

### 3. **Merge IP Geolocation**
- Converted IP addresses to integers
- Mapped IPs to countries using IP range joins
- Added `country` feature to transaction data

### 4. **Feature Engineering**
- Created:
  - Time-based: `purchase_hour`, `time_since_signup_hrs`
  - Behavior-based: `user_transaction_count`, `avg_purchase_value`, `is_first_transaction`
  - Frequency-based: `device_freq`, `browser_freq`

### 5. **Data Transformation**
- Used **SMOTE** to oversample fraud cases
- Scaled numerical features with `StandardScaler`
- One-hot encoded categorical features (e.g., `browser`, `country`)
- Created final train/test splits

---

## 🧠 Next Steps (Optional Task 6)

- Train classifiers: Logistic Regression, Random Forest, XGBoost
- Evaluate using:
  - Confusion matrix
  - Precision / Recall / F1-Score
  - ROC-AUC
- Tune for low false negatives

---

## 🛠️ Dependencies

Install required libraries with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
