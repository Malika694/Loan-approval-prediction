# 💳 Loan Approval Prediction

This project is part of a machine learning , where the objective is to predict whether a loan application will be **approved or not approved** based on applicant details.

## 📌 Project Overview

* **Objective**: Predict loan approval status from applicant and loan details.
* **Dataset**: Loan Prediction dataset (CSV file inside `.zip`).
* **Algorithm Used**: Random Forest Classifier.
* **Evaluation Metrics**: Classification Report, Confusion Matrix.

## 🛠️ Steps in the Project

1. **Import Libraries**

   * pandas, numpy, seaborn, matplotlib
   * scikit-learn (train\_test\_split, LabelEncoder, StandardScaler, RandomForestClassifier, metrics)

2. **Load the Dataset**

   * Read the loan dataset from a zipped CSV file.

3. **Data Preprocessing**

   * Dropped `Loan_ID` column (not useful for prediction).
   * Filled missing values with mode (for categorical) and median (for numerical).
   * Encoded categorical variables using `LabelEncoder`.
   * Standardized numerical features with `StandardScaler`.

4. **Feature-Target Split**

   * Features: Applicant details & loan attributes.
   * Target: `Loan_Status` (1 = Approved, 0 = Not Approved).

5. **Model Training**

   * Train-Test split (80/20).
   * Trained a **Random Forest Classifier**.

6. **Model Evaluation**

   * Classification report (precision, recall, f1-score).
   * Confusion matrix heatmap.

## 📊 Results

* The Random Forest Classifier achieved strong predictive performance.
* Evaluation results (metrics + confusion matrix) are shown in the notebook.

## 🚀 How to Run the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/loan-approval-prediction.git
   cd loan-approval-prediction
   ```

2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:

   ```bash
   jupyter notebook Task_4_Loan_Approval_Prediction_Description.ipynb
   ```

## 📦 Requirements

* Python 3.8+
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

Install them via:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```



