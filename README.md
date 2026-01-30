# 🎓 Student Dropout Prediction using Logistic Regression

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green.svg)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-blue.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-90.77%25-brightgreen.svg)

*A machine learning project to predict student dropout using logistic regression, comparing a custom implementation with scikit-learn's built-in model.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Technical Implementation](#-technical-implementation)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Author](#-author)
- [License](#-license)

---

## 🔍 Overview

This project implements a **binary classification model** to predict whether a student will **dropout** or **graduate** based on various academic, demographic, and socioeconomic features. The project demonstrates both a **from-scratch implementation** of logistic regression and a comparison with **scikit-learn's LogisticRegression**.

### 🎯 Objective
- Predict student academic outcomes (Dropout vs. Graduate)
- Implement logistic regression from scratch using gradient descent
- Validate the custom implementation against scikit-learn's model

---

## ✨ Features

- **Custom Logistic Regression Implementation**
  - Sigmoid activation function
  - Log-loss (Binary Cross-Entropy) cost function
  - Gradient descent optimization
  - Numerical stability with clipping

- **Data Preprocessing**
  - Feature standardization using StandardScaler
  - Binary target encoding (Dropout: 0, Graduate: 1)
  - Train/test split (80/20)

- **Model Comparison**
  - Custom implementation vs. scikit-learn
  - Performance validation and benchmarking

---

## 📊 Dataset

The dataset contains **3,630 student records** with **36 features** after preprocessing.

### Key Features

| Category | Features |
|----------|----------|
| **Demographics** | Marital status, Gender, Age at enrollment, Nationality |
| **Academic Background** | Previous qualification, Application mode, Course |
| **Family Background** | Mother's/Father's qualification, Mother's/Father's occupation |
| **Academic Performance** | Curricular units (1st & 2nd semester): credited, enrolled, evaluations, approved, grade |
| **Financial** | Debtor, Tuition fees up to date, Scholarship holder |
| **Socioeconomic** | Unemployment rate, Inflation rate, GDP |

### Target Variable
- **0**: Dropout
- **1**: Graduate

> **Note**: Students with "Enrolled" status were excluded to create a clear binary classification problem.

---

## 🔧 Technical Implementation

### Mathematical Foundation

**Sigmoid Function:**
```
σ(z) = 1 / (1 + e^(-z))
```

**Model:**
```
h(X) = σ(X · θ)
```

**Cost Function (Binary Cross-Entropy):**
```
J(θ) = -1/m Σ [y·log(h(X)) + (1-y)·log(1-h(X))]
```

**Gradient Descent Update:**
```
θ = θ - α · (1/m) · X^T · (h(X) - y)
```

### Hyperparameters
- **Learning Rate (α)**: 0.1
- **Iterations**: 6,000
- **Train/Test Split**: 80% / 20%

### Data Preprocessing Pipeline
1. Load data from CSV with semicolon separator
2. Filter out "Enrolled" students
3. Encode target: Dropout → 0, Graduate → 1
4. Split into train/test sets
5. Apply StandardScaler normalization
6. Add bias column for gradient descent

---

## 📈 Results

### Model Performance

| Metric | Custom Model | scikit-learn |
|--------|-------------|--------------|
| **Accuracy** | 90.77% | 90.77% |

### Training Convergence
The custom implementation shows smooth convergence of the cost function over 6,000 iterations, demonstrating successful optimization.

![Cost Function Convergence](cost_convergence.png)

> The identical performance validates our custom implementation correctly replicates scikit-learn's logistic regression behavior.

---

## 🛠 Installation

### Prerequisites
- Python 3.12+
- Jupyter Notebook

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/student-dropout-prediction.git
cd student-dropout-prediction

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

## 🚀 Usage

### Running the Notebook

```bash
jupyter notebook Logistic_Regression_Students_Dropout.ipynb
```

### Quick Start

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv("data.csv", sep=";")

# Filter and encode
data = data[data["Target"] != "Enrolled"]
data.Target.replace(["Dropout", "Graduate"], [0, 1], inplace=True)

# Prepare features and target
X = data.drop(["Target"], axis=1).values
y = data.Target.values

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

## 📁 Project Structure

```
student-dropout-prediction/
│
├── 📓 Logistic_Regression_Students_Dropout.ipynb   # Main notebook
├── 📊 data.csv                                      # Dataset
├── 📖 README.md                                     # This file
├── 📋 requirements.txt                              # Dependencies
└── 🖼️ cost_convergence.png                          # Training visualization
```

---

## 🛠 Technologies Used

<div align="center">

| Technology | Purpose |
|------------|---------|
| ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white) | Programming Language |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat&logo=numpy&logoColor=white) | Numerical Computing |
| ![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat&logo=pandas&logoColor=white) | Data Manipulation |
| ![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | Machine Learning |
| ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557c?style=flat) | Visualization |
| ![Seaborn](https://img.shields.io/badge/-Seaborn-3776AB?style=flat) | Statistical Visualization |
| ![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) | Interactive Development |

</div>

---

## 🔮 Future Improvements

- [ ] Feature importance analysis
- [ ] Cross-validation implementation
- [ ] Additional metrics (Precision, Recall, F1-Score, ROC-AUC)
- [ ] Hyperparameter tuning (learning rate, regularization)
- [ ] Comparison with other algorithms (Random Forest, SVM, Neural Networks)
- [ ] Early stopping mechanism
- [ ] SHAP values for model interpretability

---

## 👤 Author

**Samah Aziz**

- 🎓 Licence Ingénierie Informatique Student
- 💼 Passionate about Machine Learning & Data Science
- 🔗 [LinkedIn](https://linkedin.com/in/yourprofile)
- 💻 [GitHub](https://github.com/yourusername)

---

<div align="center">

⭐ **If you found this project helpful, please give it a star!** ⭐

Made with ❤️ and Python

</div>
