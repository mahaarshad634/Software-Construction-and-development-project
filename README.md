# 🤖 Auto Machine Learning (AutoML Trainer System)

An intelligent **AutoML system in Python** that automatically detects the machine learning problem type, preprocesses data, trains multiple models, evaluates them, and selects the best-performing model.

---

## 📌 Project Overview

This project automates the full machine learning workflow:

- 📂 Load dataset from CSV file
- 🧹 Clean missing values
- 🧠 Detect problem type (Classification / Regression)
- ⚙️ Preprocess data automatically
- 🤖 Train multiple ML models
- 📊 Evaluate performance
- 🏆 Select best model automatically

---

## ⚙️ Features

### 🔍 Smart Detection
- Automatically detects:
  - Classification problem
  - Regression problem

### 🧹 Data Cleaning
- Handles missing values (mean imputation)
- Converts categorical variables using encoding
- Removes target leakage automatically

### ⚙️ Preprocessing
- One-hot encoding for categorical features
- Label encoding for classification targets
- Feature scaling using StandardScaler

### 🤖 Machine Learning Models

#### Classification Models:
- Random Forest Classifier
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes

#### Regression Models:
- Random Forest Regressor
- Linear Regression
- Ridge Regression
- Decision Tree Regressor
- KNN Regressor



