# 🚢 Titanic Survival Prediction (Machine Learning + Streamlit)

## 📌 Project Overview

This project predicts whether a passenger survived the Titanic disaster using Machine Learning models.
It covers the complete ML pipeline including data preprocessing, feature engineering, model building, evaluation, and deployment using Streamlit.

---

## 🎯 Objective

To build a predictive model that determines the survival of passengers based on features such as age, gender, ticket class, fare, and family details.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib and Seaborn
* Streamlit

---

## 📊 Machine Learning Models Used

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

---

## ⚙️ Steps Involved

### 1. Data Preprocessing

* Handled missing values (Age → median, Cabin → dropped)
* Removed irrelevant columns (PassengerId, Name, Ticket)

### 2. Feature Engineering

* Created new feature: **FamilySize = SibSp + Parch**

### 3. Encoding

* Label Encoding for binary variables (Sex)
* One-Hot Encoding for categorical variables (Embarked)

### 4. Feature Scaling

* Applied StandardScaler to normalize numerical features

### 5. Model Training

* Split data into training and testing sets
* Trained multiple ML models for comparison

### 6. Model Evaluation

* Evaluated using Accuracy Score and Confusion Matrix

---

## 📈 Results

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 80%      |
| Decision Tree       | 72%      |
| Random Forest       | 83%      |

👉 **Best Model: Random Forest Classifier**

---

## 🔍 Confusion Matrix Insights

* High accuracy in predicting non-survivors
* Some survivors were misclassified (False Negatives)
* Balanced performance with good precision (~83%)

---

## 🚀 Streamlit App

An interactive web application is built using Streamlit where users can input passenger details and get real-time survival predictions.

👉 **Live App Link:**
https://titanic-survival-prediction-app-app-peiezr2vccatlbnamgcufs.streamlit.app/

---

## 📁 Project Structure

```
Titanic_Project/
│
├── titanic_notebook.ipynb
├── app.py
├── titanic_model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run Locally

1. Clone the repository:

```
git clone <your-repo-link>
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the Streamlit app:

```
streamlit run app.py
```

---

## 💡 Key Learnings

* Importance of data preprocessing in ML
* Handling categorical variables and missing data
* Model comparison and evaluation techniques
* Deploying ML models using Streamlit

---

## 👤 Author

Himanshu Bhandari | Data Analyst Professional

---
