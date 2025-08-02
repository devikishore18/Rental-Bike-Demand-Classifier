# Rental-Bike-Demand-Classifier
This project focuses on building a robust classification ML model to predict the demand level of bike rentals—categorized as Low, Medium, or High,enabling better resource planning, demand forecasting, and operational optimization for bike-sharing services.

# 🚲 Bike Demand Classification

A machine learning project to classify **bike rental demand levels** (Low, Medium, High) based on historical and environmental features using classification algorithms. This project uses the UCI Bike Sharing Dataset and implements a complete ML pipeline including EDA, preprocessing, feature selection, model training, and hyperparameter tuning.

---

## 📌 Problem Statement

Bike sharing systems offer an efficient and eco-friendly alternative to traditional transportation. However, predicting the level of demand at a given hour is crucial for proper bike availability, station management, and customer satisfaction. The goal of this project is to build a classification model that predicts bike rental demand levels—**Low**, **Medium**, or **High**—based on environmental and temporal features.

---

## 📊 Dataset Description

- **Source**: [UCI Machine Learning Repository – Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)
- **Records**: 17,379 hourly records
- **Target Variable**: `cnt_class` (categorized count)
- **Features Include**:
  - **Time Features**: Hour, Month, Year, Season
  - **Weather Features**: Temperature, Humidity, Windspeed, Weather Situation
  - **Calendar Features**: Holiday, Working Day
  - One-hot encoded month and season indicators

---

## 🔬 Data Preprocessing

- Label binning: Target `count` converted to categories (Low, Medium, High)
- Dropped leakage-prone variables: `casual`, `registered`, `count`
- Used SMOTE to address class imbalance.
- Feature selection: Used `SelectKBest` to select top features
- Scaling: Applied `StandardScaler` to numeric features
- Categorical encoding: One-hot encoded months, seasons, and year

---

## 🧠 Model Training & Evaluation

Several models were trained and evaluated on the dataset:

| Model                 | Accuracy |
|----------------------|----------|
| Logistic Regression  | 0.63     |
| Decision Tree        | 0.84     |
| Random Forest        | 0.86 ✅ |
| AdaBoost             | 0.65     |
| Gradient Boosting    | 0.83     |

✅ **Random Forest** provided the best performance after hyperparameter tuning.

---

## ⚙️ Hyperparameter Tuning

- GridSearchCV was used to optimize Random Forest hyperparameters.
- Improved model slightly with best estimator settings
- Tuned Random Forest Evaluation: Accuracy: 0.8637763563136698
---

## 🧾 Evaluation Metrics

- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1 Score**  
- **Confusion Matrix**  
- **Classification Report**

---

## 💾 Pipeline Deployment and Joblib Saving

Before predicting on new/unseen data:
- A complete pipeline was constructed combining preprocessing steps and the best-performing classifier.
- The pipeline was saved using `joblib` for reusability and fast deployment.
- The saved pipeline can be directly loaded and used for prediction on new data without repeating preprocessing or training.

---

## 🔮 Predicting Unseen Data

```python
import pandas as pd

unseen_data = pd.DataFrame([{
    'instant': 1,
    'hr': 0,
    'weathersit': 1.0,
    'temp': 0.24,
    'atemp': 0.2879,
    'hum': 0.81,
    'windspeed': 0.1,  # Add if used during training
    'yr_2012': 0,
    'mnth_jan': 1,
    'mnth_aug': 0,
    'mnth_dec': 0,
    'mnth_feb': 0,
    'mnth_jul': 0,
    'mnth_jun': 0,
    'mnth_mar': 0,
    'mnth_sep': 0,
    'workingday_workday': 1,
    'season_spring': 0,
    'season_summer': 0,
    'season_winter': 1
}])
```
### Predictions on Unseen Data
Tested the model with a new data point

### OUTPUT

**Predicted Bike Count Class: High**

## 🛠 Tech Stack
### 🔢 Languages & Libraries
- Python – Core programming language used
- NumPy – Numerical operations and array handling
- Pandas – Data manipulation and preprocessing
- Matplotlib & Seaborn – Data visualization and exploratory analysis
- imblearn (imbalanced-learn) – Handling class imbalance (e.g., SMOTE, RandomOverSampler)
- Scikit-learn – Machine learning models, preprocessing, pipelines, and evaluation
- Joblib – Model serialization and deployment

### Machine Learning & Preprocessing
- Models: Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting
- Resampling: SMOTE (from imblearn)
- Scaler: StandardScaler
- Pipeline: sklearn.pipeline
- Model Evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Classification Report

### 🖥️ Tools & Environment
- Jupyter Notebook – Interactive development environment
- VS Code or any Python IDE – for editing and experimentation
- Git & GitHub – Version control and project sharing

---

## ✅ Conclusion

This project successfully demonstrates a complete machine learning pipeline for classifying bike rental demand levels (Low, Medium, High) using the UCI Bike Sharing Dataset. 

Through thorough exploratory data analysis (EDA), careful preprocessing, and feature selection using `SelectKBest`, we identified key factors affecting bike demand. Models such as Logistic Regression, Decision Trees, Random Forest, AdaBoost, and Gradient Boosting were evaluated. After hyperparameter tuning, the **Random Forest classifier** emerged as the best performer with an accuracy of **86%**.

A scalable `Pipeline` was created and saved using `joblib`, enabling easy deployment and prediction on unseen data. The project demonstrates the value of classification models in predicting bike rental behavior based on weather, time, and seasonal data—offering insights that could help improve operational efficiency in real-world bike-sharing systems.

Future work can include deploying this model as a web application and integrating real-time data streams to support dynamic demand forecasting.

---


## **📞 Contact**
For questions, feedback, or collaboration, feel free to reach out:
Email: devikishore18@gmail.com
LinkedIn: https://www.linkedin.com/in/devikishore18/

