# 📊 CHURN.APP – Customer Churn Prediction

A machine learning-powered web app that predicts whether a customer is likely to churn. Built with Python, Streamlit, and scikit-learn, this project demonstrates real-time classification using structured input data, probability scores, and sample testing.

---

## 📽️ Demo Video

[Click to watch the demo](https://raw.githubusercontent.com/Bhoomika08-MAY/CODSOFT-2/main/assets/demo.mp4)

---

## 🧠 Features

- 🔍 Real-time churn prediction using a trained classification model
- 📊 Probability score for prediction confidence
- 📝 Sample data loading via CSV or dropdowns
- 📁 Clean UI with organized assets and modular code structure

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- scikit-learn  
- pandas  
- Git & GitHub

---

## 📂 Project Structure

| 📁 Folder/File           | 📝 Description                                                  |
|-------------------------|-----------------------------------------------------------------|
| `app.py`                | Main Streamlit app for churn prediction UI and logic           |
| `churn.model`           | Trained machine learning model saved in pickle format          |
| `dummy.csv`             | Sample customer data for testing predictions                   |
| `requirements.txt`      | List of Python dependencies for easy setup                     |
| `.gitignore`            | Specifies files/folders to exclude from Git tracking           |
| `README.md`             | Project overview, demo link, and usage instructions            |
| `assets/demo.mp4`       | Demo video showcasing the app in action                        |

---

## 📦 Installation

```bash
pip install -r requirements.txt
streamlit run app.py
