# Employee Attrition Prediction – Ipsha Gautam

This project implements a **machine learning pipeline** in Python using **Scikit-learn** to predict employee attrition.  
The dataset used is the well-known **IBM HR Employee Attrition dataset** (1,470 employee records).  

## ✨ Key Features
- Preprocessed HR dataset: handled categorical encoding, normalization, and missing values.
- Built and compared models: Logistic Regression vs Random Forest Classifier.
- Evaluated performance with accuracy, precision, recall, and confusion matrix.
- Visualized key feature importances to understand drivers of attrition.

## 📂 Project Structure
```
employee-attrition-prediction/
├─ notebooks/employee_attrition.ipynb
├─ scripts/employee_attrition.py
├─ sample_data/WA_Fn-UseC_-HR-Employee-Attrition.csv
├─ outputs/ (confusion matrix, feature importances)
├─ requirements.txt
├─ README.md
├─ .gitignore
└─ LICENSE
```

## 🚀 How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the notebook in Jupyter: `notebooks/employee_attrition.ipynb`
3. Or run the script: `python scripts/employee_attrition.py`
