# Intern Portfolio Readiness Scorer

## Overview
Intern Portfolio Readiness Scorer is an AI/ML-based system designed to evaluate candidate employability through GitHub portfolio analysis. The system automatically extracts GitHub profile information either from uploaded resumes or direct GitHub links and predicts whether a candidate is:

- Job Ready
- Almost Ready
- Needs Improvement

The project uses Machine Learning, GitHub REST API, feature engineering, and an interactive Streamlit dashboard for portfolio evaluation and analytics.

---

# Features

- Resume Upload (PDF/DOCX)
- Automatic GitHub Link Extraction
- GitHub REST API Integration
- Portfolio Feature Extraction
- Machine Learning Prediction
- Professional Dashboard Analytics
- Personalized Recommendations
- Dataset Visualization
- Portfolio Readiness Scoring

---

# Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- GitHub REST API
- Joblib

---

# Project Workflow

Resume Upload / GitHub Link  
↓  
GitHub Link Extraction  
↓  
GitHub REST API Feature Extraction  
↓  
Feature Engineering  
↓  
Machine Learning Prediction  
↓  
Dashboard & Recommendations

---

# Folder Structure

Project/
│
├── app.py
├── github_api.py
├── resume_extractor.py
├── train_model_advanced.py
│
├── models/
│   ├── best_readiness_model.pkl
│   ├── label_encoder.pkl
│   └── model_features.json
│
├── data/
│   └── final_portfolio_dataset.csv
│
├── .streamlit/
│   └── secrets.toml
│
└── requirements.txt

---

# Installation Steps

## 1. Clone or Extract Project

Extract the ZIP file or clone the repository.

---

## 2. Install Required Libraries

Run the following command:

   
pip install -r requirements.txt
 

If requirements.txt is unavailable, install manually:

   
pip install streamlit pandas numpy scikit-learn plotly requests PyPDF2 python-docx imbalanced-learn joblib
 

---

# GitHub API Setup

Create file:

.streamlit/secrets.toml
 

Add your GitHub token:

GITHUB_TOKEN = "your_github_token_here"
 

---

# Running the Project

Run the Streamlit application:

   
streamlit run app.py
 

The application will automatically open in the browser.

---

# Using the Application

## Option 1: Upload Resume
- Upload PDF or DOCX resume
- System automatically extracts GitHub link
- GitHub portfolio gets analyzed
- ML model predicts readiness category

## Option 2: GitHub Profile Link
- Enter GitHub profile link directly
- System extracts portfolio features
- Prediction and analytics are displayed

---

# Machine Learning Details

Models Used:
- Logistic Regression
- Decision Tree
- Random Forest
- SVM

Final Selected Model:
- Logistic Regression

Model Accuracy:
- Approximately 95%

Class balancing was handled using SMOTE to improve prediction fairness across all readiness categories.

---

# Dashboard Features

- Readiness Prediction
- Portfolio Score
- GitHub Portfolio Summary
- Skill Gap Analysis
- Score Breakdown Charts
- Dataset Analytics
- Personalized Recommendations

---

# Output Categories

- Job Ready
- Almost Ready
- Needs Improvement

---

# Future Scope

- LinkedIn Portfolio Analysis
- Multi-platform Portfolio Evaluation
- Resume Ranking System
- Real-time Recruiter Dashboard
- Cloud Deployment

---

# Team

Team DSA3

---

# Author

Parth Shendre

