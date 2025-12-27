# expnense_tracking_project
this is my project of developing a simple and basic expense tracking system for the personal finance.this uses NLP to handle the text and autotag it to the correct category.
autotagger-expense/
│
├── data/
│   └── expenses.csv                # Sample dataset
│
├── notebooks/
│   └── EDA_and_Modeling.ipynb     # Exploratory analysis and model training
│
├── src/
│   ├── preprocess.py              # Data cleaning and encoding
│   ├── train_model.py             # Model training script
│   ├── predict.py                 # Prediction logic
│   └── utils.py                   # Helper functions
│
├── api/
│   └── app.py                     # Flask API
│
├── tests/
│   └── test_api.py                # API tests
│
├── model/
│   └── rf_model.pkl               # Serialized trained model
│
├── requirements.txt
└── README.md


#explanation of the previous structured overview

# 🧾 AutoTagger: Expense Categorization with ML & Flask API

Automatically categorize personal or business expenses using a machine learning pipeline built with scikit-learn and deployed via a lightweight Flask API.

## 🚀 Project Overview

This project demonstrates an end-to-end machine learning pipeline for **automated expense categorization**, from raw data preprocessing to model training and deployment as a RESTful API. It’s designed for personal finance apps, startups, or freelancers looking to integrate intelligent expense tagging into their systems.

## 🧠 Features

- 🔍 **Data Preprocessing**: Handles missing values, encodes categorical variables using `scikit-learn` encoders.
- 🌲 **Modeling**: Trains a `RandomForestClassifier` for robust, interpretable classification.
- 🧪 **Evaluation**: Includes metrics like accuracy, precision, recall, and confusion matrix.
- 🌐 **Deployment**: Exposes the trained model via a Flask API for real-time predictions.
- 🧪 **Test Suite**: Includes unit tests for preprocessing and API endpoints.

## 📁 Project Structure

