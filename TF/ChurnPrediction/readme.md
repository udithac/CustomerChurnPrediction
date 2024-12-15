# Churn Prediction Project

This project predicts customer churn using a machine learning model built with TensorFlow and scikit-learn. The model identifies customers who are likely to stop using a service based on their demographic, usage, and contract-related data.

---

## 📋 Features
- Preprocesses customer data (categorical and numerical).
- Trains a Neural Network using TensorFlow.
- Predicts churn probability for new customers.
- Includes a fully functional Python script for model training and prediction.

---

## 🛠️ Technologies Used
- **Programming Language**: Python
- **Machine Learning Framework**: TensorFlow, Keras
- **Data Processing**: scikit-learn, pandas
- **Development Environment**: VS Code
- **Version Control**: Git

---

## 📁 Project Structure
```plaintext
ChurnPrediction/
├── data/                     # Data folder (if applicable)
│   └── customerdata.csv      # Example dataset (not included by default)
├── model/                    # Model folder
│   └── cs_churn_tfmodel.keras # Trained model file
├── churn_prediction.py       # Main script for predictions
├── churn_training.py         # Script for training the model
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # Project documentation
