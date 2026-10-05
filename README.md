# Housing Price Prediction

A Machine Learning regression project that predicts house prices based on property characteristics such as area, bedrooms, bathrooms, stories, parking, and various amenities.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Objectives](#objectives)
* [Dataset](#dataset)
* [Dataset Features](#dataset-features)
* [Technologies Used](#technologies-used)
* [Project Workflow](#project-workflow)
* [Data Preprocessing](#data-preprocessing)
* [Exploratory Data Analysis](#exploratory-data-analysis)
* [Feature Engineering](#feature-engineering)
* [Model Development](#model-development)
* [Model Evaluation](#model-evaluation)
* [Results](#results)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Requirements](#requirements)
* [Future Improvements](#future-improvements)
* [Conclusion](#conclusion)
* [Author](#author)

---

## Project Overview

House price prediction is a regression problem where the objective is to estimate the price of a property based on its characteristics.

This project uses a **Linear Regression** model to predict house prices from numerical and categorical property features.

The complete machine learning workflow includes:

1. Loading the dataset
2. Understanding the data
3. Exploratory Data Analysis
4. Detecting and removing outliers
5. Encoding categorical variables
6. Selecting features
7. Splitting the dataset
8. Training the regression model
9. Making predictions
10. Evaluating model performance

---

## Objectives

The main objectives of this project are:

* Analyze housing price data.
* Understand the relationship between property features and price.
* Identify and handle outliers.
* Convert categorical features into numerical values.
* Train a regression model for price prediction.
* Evaluate the model using standard regression metrics.
* Establish a baseline model that can be improved using advanced algorithms.

---

## Dataset

The dataset contains information about residential properties and their corresponding prices.

The target variable is:

```text
price
```

The model uses property-related features such as:

* Area
* Number of bedrooms
* Number of bathrooms
* Number of stories
* Parking availability
* Main road access
* Guestroom availability
* Basement availability
* Hot water heating
* Air conditioning
* Preferred area
* Furnishing status

---

## Dataset Features

| Feature            | Description                                 | Type        |
| ------------------ | ------------------------------------------- | ----------- |
| `area`             | Area of the house in square feet            | Numerical   |
| `bedrooms`         | Number of bedrooms                          | Numerical   |
| `bathrooms`        | Number of bathrooms                         | Numerical   |
| `stories`          | Number of stories                           | Numerical   |
| `parking`          | Number of parking spaces                    | Numerical   |
| `mainroad`         | Whether the property has main road access   | Categorical |
| `guestroom`        | Whether a guestroom is available            | Categorical |
| `basement`         | Whether a basement is available             | Categorical |
| `hotwaterheating`  | Whether hot water heating is available      | Categorical |
| `airconditioning`  | Whether air conditioning is available       | Categorical |
| `prefarea`         | Whether the property is in a preferred area | Categorical |
| `furnishingstatus` | Furnishing status of the property           | Categorical |
| `price`            | House price                                 | Target      |

---

## Categorical Feature Encoding

The categorical features are converted into numerical values before model training.

### Binary Features

Features such as:

```text
mainroad
guestroom
basement
hotwaterheating
airconditioning
prefarea
```

are represented using binary values:

```text
Yes → 1
No  → 0
```

### Furnishing Status

The `furnishingstatus` feature is converted into numerical labels:

```text
Unfurnished      → 0
Semi-furnished   → 1
Furnished        → 2
```

---

## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

### Development Environment

* Jupyter Notebook
* Google Colab
* VS Code

### Machine Learning

* Linear Regression
* Train-Test Split
* Regression Evaluation Metrics

---

## Project Workflow

```text
Dataset
   |
   v
Data Loading
   |
   v
Data Understanding
   |
   v
Exploratory Data Analysis
   |
   v
Outlier Detection
   |
   v
Outlier Removal
   |
   v
Categorical Encoding
   |
   v
Feature Selection
   |
   v
Train-Test Split
   |
   v
Linear Regression
   |
   v
Prediction
   |
   v
Model Evaluation
   |
   v
Results
```

---

## Data Preprocessing

### 1. Data Loading

The dataset is loaded using Pandas.

```python
import pandas as pd

df = pd.read_csv("Housing.csv")
```

---

### 2. Data Inspection

The dataset is inspected to understand its structure, data types, missing values, and statistical characteristics.

Common operations include:

```python
df.head()
df.info()
df.describe()
df.isnull().sum()
```

---

### 3. Missing Value Analysis

Missing values are checked before model training.

```python
df.isnull().sum()
```

If missing values are present, they should be handled appropriately before proceeding with model training.

---

## Exploratory Data Analysis

Exploratory Data Analysis is performed to understand the distribution of the data and relationships between different features.

### Price Distribution

The distribution of house prices can be visualized using:

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(df["price"], kde=True)
plt.title("House Price Distribution")
plt.show()
```

### Area vs Price

The relationship between house area and price can be visualized using:

```python
sns.scatterplot(data=df, x="area", y="price")
plt.title("Area vs House Price")
plt.show()
```

This helps identify whether larger houses generally have higher prices.

---

## Outlier Detection

Outliers can significantly influence regression models.

The **Interquartile Range (IQR)** method is used to identify extreme values.

The IQR is calculated as:

```text
IQR = Q3 - Q1
```

The lower and upper boundaries are:

```text
Lower Bound = Q1 - 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR
```

Values outside these boundaries are considered potential outliers.

In this project, outlier handling is applied to important numerical variables such as:

```text
area
price
```

This helps reduce the influence of extreme observations.

---

## Feature Engineering

Feature engineering involves preparing the available variables so that they can be effectively used by the machine learning model.

The final feature set includes:

```python
features = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking",
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus"
]
```

The target variable is:

```python
target = "price"
```

---

## Train-Test Split

The dataset is divided into training and testing sets.

```python
from sklearn.model_selection import train_test_split

X = df[features]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

### Dataset Split

```text
Training Data → 80%
Testing Data  → 20%
```

The training dataset is used to learn the relationship between the input features and house prices.

The testing dataset is used to evaluate how well the trained model performs on unseen data.

---

# Model Development

## Linear Regression

Linear Regression is used as the baseline machine learning algorithm.

The model attempts to learn a linear relationship between the input features and the target variable.

The general equation is:

```text
y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ
```

Where:

* `y` = predicted house price
* `β₀` = intercept
* `β₁ ... βₙ` = model coefficients
* `X₁ ... Xₙ` = input features

---

## Model Training

The Linear Regression model can be trained using Scikit-learn:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
```

---

## Making Predictions

After training, predictions are generated using the testing data:

```python
y_pred = model.predict(X_test)
```

The predicted values are then compared with the actual house prices.

---

# Model Evaluation

Three standard regression metrics are used to evaluate the model:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* R² Score

---

## Mean Absolute Error

MAE measures the average absolute difference between actual and predicted values.

Formula:

```text
MAE = (1/n) × Σ |yᵢ - ŷᵢ|
```

Where:

* `yᵢ` = actual value
* `ŷᵢ` = predicted value
* `n` = number of observations

A lower MAE indicates better performance.

Python implementation:

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
```

---

## Mean Squared Error

MSE calculates the average squared difference between actual and predicted values.

Formula:

```text
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²
```

Because the errors are squared, larger prediction errors have a greater impact on the metric.

Python implementation:

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
```

A lower MSE indicates better performance.

---

## R² Score

The R² score measures the proportion of variance in the target variable that is explained by the model.

Formula:

```text
R² = 1 - SSres / SStot
```

Where:

* `SSres` = residual sum of squares
* `SStot` = total sum of squares

Python implementation:

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
```

An R² value closer to `1` generally indicates a better fit.

---

# Results

Example model performance:

```text
MAE: 741836.30
MSE: 9.56e11
R² Score: 0.69
```

### Performance Summary

| Metric   |      Result |
| -------- | ----------: |
| MAE      |  741,836.30 |
| MSE      | 9.56 × 10¹¹ |
| R² Score |        0.69 |

The Linear Regression model achieved an R² score of approximately **0.69**, meaning that approximately **69% of the variance in house prices is explained by the model on the evaluated test set**.

The MAE indicates the average absolute difference between the actual and predicted house prices.

---

# Prediction Visualization

Actual and predicted house prices can be compared visually using a scatter plot.

```python
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()
```

A well-performing model will generally have predictions concentrated around the ideal:

```text
Predicted Price = Actual Price
```

---

# Actual vs Predicted Values

A comparison table can also be created:

```python
comparison = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": y_pred
})

print(comparison.head(10))
```

This allows individual predictions to be compared with their actual values.

---

# Project Structure

```text
housing-price-prediction/
│
├── dataset/
│   └── Housing.csv
│
├── main.py
│
├── requirements.txt
│
├── README.md
│
└── notebooks/
    └── housing_price_prediction.ipynb
```

---

# Installation

## Step 1: Clone the Repository

```bash
git clone <your-repository-url>
```

Navigate into the project directory:

```bash
cd housing-price-prediction
```

---

## Step 2: Create a Virtual Environment

Creating a virtual environment is recommended.

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Step 3: Install Dependencies

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

---

# Requirements

Example `requirements.txt`:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

# Usage

After installing the dependencies, run the project using:

```bash
python main.py
```

If you are using Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
housing_price_prediction.ipynb
```

Run the notebook cells sequentially.

---

# Example Prediction

After training the model, a new property's features can be passed to the model for prediction.

Example:

```python
new_house = [[
    5000,   # area
    3,      # bedrooms
    2,      # bathrooms
    2,      # stories
    2,      # parking
    1,      # mainroad
    1,      # guestroom
    1,      # basement
    0,      # hotwaterheating
    1,      # airconditioning
    1,      # prefarea
    1       # furnishingstatus
]]

predicted_price = model.predict(new_house)

print("Predicted House Price:", predicted_price[0])
```

The feature order must match the order used during model training.

---

# Key Learnings

This project demonstrates practical knowledge of:

* Regression problems
* Data preprocessing
* Exploratory Data Analysis
* Outlier detection
* IQR-based outlier removal
* Categorical feature encoding
* Feature selection
* Train-test splitting
* Linear Regression
* Model prediction
* MAE
* MSE
* R² Score
* Data visualization
* Python-based machine learning workflows

---

# Future Improvements

The baseline Linear Regression model can be improved through additional experimentation.

### 1. Advanced Regression Models

Experiment with:

* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost
* Decision Tree Regressor
* Support Vector Regression

### 2. Feature Scaling

Apply techniques such as:

```text
StandardScaler
MinMaxScaler
```

where appropriate.

### 3. Cross-Validation

Use K-Fold Cross-Validation to obtain a more robust estimate of model performance.

```text
K-Fold Cross-Validation
        |
        +-- Fold 1
        +-- Fold 2
        +-- Fold 3
        +-- Fold 4
        +-- Fold 5
```

### 4. Hyperparameter Tuning

Use:

* GridSearchCV
* RandomizedSearchCV

to optimize advanced machine learning models.

### 5. Log Transformation

House prices can be skewed. Applying a logarithmic transformation to the target variable may improve model performance:

```python
import numpy as np

y_log = np.log1p(y)
```

### 6. Model Comparison

Compare multiple algorithms using the same evaluation metrics:

| Model             | MAE | MSE | R² |
| ----------------- | --: | --: | -: |
| Linear Regression |   - |   - |  - |
| Decision Tree     |   - |   - |  - |
| Random Forest     |   - |   - |  - |
| Gradient Boosting |   - |   - |  - |
| XGBoost           |   - |   - |  - |

This makes it easier to identify the best-performing model.

### 7. Deployment

The trained model can be deployed using:

* Flask
* FastAPI
* Streamlit

A web interface can then allow users to enter property details and receive a predicted house price.

---

# Limitations

The current implementation has several limitations:

* Linear Regression assumes a linear relationship between features and the target.
* The dataset may not represent all housing markets.
* Extreme values can affect regression performance.
* Label encoding of `furnishingstatus` introduces an ordinal numerical representation that may not fully represent the categorical relationship.
* Model performance depends on the quality and distribution of the available dataset.
* The current model is a baseline and has not been extensively hyperparameter-tuned.

---

# Conclusion

The **Housing Price Prediction** project demonstrates an end-to-end machine learning regression workflow.

The project starts with raw housing data and performs data exploration, preprocessing, outlier handling, categorical encoding, feature selection, and train-test splitting. A **Linear Regression model** is then trained to predict house prices.

The model is evaluated using **MAE, MSE, and R² Score**, achieving an example R² score of **0.69**.

This project provides a strong baseline for further experimentation with advanced regression algorithms, feature engineering, cross-validation, hyperparameter tuning, and model deployment.

---

# Author

**Inzeera Z**

B.Tech (Hons) Artificial Intelligence and Data Science

Arunachala College of Engineering for Women

### Areas of Interest

* Artificial Intelligence
* Machine Learning
* Deep Learning
* Generative AI
* Computer Vision
* Data Science
* AI Engineering

---
