# 🛡️ Fraud Detection System – Adey Innovations Inc.

## 📌 Project Overview
This project aims to develop machine learning models for detecting fraudulent transactions in both **e-commerce** and **banking** systems. It combines **behavioral features**, **time-based patterns**, and **IP geolocation** to enhance detection accuracy.  
We build, train, and evaluate models on two different datasets, optimizing for imbalanced data scenarios.

---

## 📂 Datasets
| File | Description |
|------|-------------|
| **Fraud_Data.csv** | E-commerce transactions with fraud labels and features like time, device, etc. |
| **IpAddress_to_Country.csv** | Maps numeric IP address ranges to countries |
| **creditcard.csv** (zipped) | Bank transactions with anonymized PCA-transformed variables (V1 to V28), amount, and class |

---

## ✅ Tasks Completed

### **Task 1 – Data Analysis & Preprocessing**
1. **Data Loading & Cleaning**
   - Loaded datasets using pandas
   - Handled missing values and duplicates
   - Converted timestamps to datetime format

2. **Exploratory Data Analysis (EDA)**
   - Visualized fraud rates by browser, hour, and country
   - Checked class imbalance (severe in bank data, moderate in e-commerce)

3. **Merge IP Geolocation**
   - Converted IP addresses to integers
   - Mapped IPs to countries using IP range joins
   - Added country feature to transaction data

4. **Feature Engineering**
   - **Time-based**: `purchase_hour`, `time_since_signup_hrs`
   - **Behavior-based**: `user_transaction_count`, `user_avg_purchase_value`, `is_first_transaction`
   - **Frequency-based**: `device_freq`, `browser_freq`

5. **Data Transformation**
   - Used **SMOTENC** to oversample fraud cases in e-commerce data
   - Scaled numerical features with **StandardScaler**
   - One-hot encoded categorical features (e.g., `browser`, `country`)
   - Created final **train/test splits** for modeling

---

### **Task 2 – Model Building & Training**
1. **Data Preparation**
   - Used **train_test_split** with stratification to preserve fraud ratio
   - For `Fraud_Data` → kept preprocessing pipeline from Task 1
   - For `creditcard.csv` → scaled all numeric features using StandardScaler

2. **Model Selection**
   - **Logistic Regression**: Simple, interpretable baseline using `class_weight="balanced"`
   - **XGBoost Classifier**: Powerful gradient boosting ensemble with `scale_pos_weight` for imbalance handling

3. **Model Training**
   - Trained both models separately on each dataset:
     - Logistic Regression (E-Commerce)
     - XGBoost (E-Commerce)
     - Logistic Regression (Bank Transactions)
     - XGBoost (Bank Transactions)

4. **Evaluation Metrics**
   - **AUC-PR** (main metric for imbalanced data)
   - **ROC-AUC**
   - **Accuracy** & **Precision** at:
     - Default threshold `0.5`
     - Best-F1 threshold (automatically determined)
   - **F1-score**
   - **Confusion Matrix** for both thresholds

5. **Model Comparison**
   - Implemented `pick_best()` function to:
     - Compare models by AUC-PR, ROC-AUC, Accuracy, Precision
     - Show metrics at both thresholds
   - Found **XGBoost** generally outperforms Logistic Regression in AUC-PR, especially in precision at optimized thresholds

---


## 🛠️ Dependencies
Install required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
