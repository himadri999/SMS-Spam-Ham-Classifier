# 📱 SMS Spam/Ham Classifier

A Machine Learning project that classifies text messages as **Spam** or **Ham (Not Spam)** using **Natural Language Processing (NLP)** and a **Naive Bayes Classifier**.  
Built with **Python**, **Scikit-learn**, and an interactive **Streamlit web app**.

---

## 🧠 Overview

This project predicts whether an incoming SMS message is spam or not.  
It uses the **SMS Spam Collection Dataset**, applies **text preprocessing** techniques like tokenization and TF-IDF vectorization, and trains a **Multinomial Naive Bayes** classifier to make predictions.

The goal is to demonstrate how text classification models can identify unwanted or fraudulent messages based on their content.

---

## 📊 Dataset Details

- **Dataset Name:** SMS Spam Collection Dataset  
- **Source:** UCI Machine Learning Repository  
- **Data Size:** ~5,500 SMS messages  
- **Classes:**  
  - `ham` → Non-spam (legitimate) messages  
  - `spam` → Unwanted/promotional/fraud messages  

Each record contains:
Label (spam/ham), Message text

Example:
ham, "Hey, are we still meeting today?"
spam, "Congratulations! You've won a free iPhone! Click here to claim."


---

## 🧮 Model & Methodology

| Step | Description |
|------|--------------|
| **1. Data Cleaning** | Removed punctuation, symbols, and stopwords |
| **2. Text Vectorization** | Used TF-IDF (Term Frequency–Inverse Document Frequency) to convert text into numerical form |
| **3. Model Training** | Applied **Multinomial Naive Bayes** for probabilistic classification |
| **4. Model Saving** | Trained model was saved using `joblib` for deployment |
| **5. Web Interface** | Built using Streamlit to allow user input and live predictions |

---

## ⚙️ Technologies Used

- **Python**
- **Streamlit** (for the web app)
- **Scikit-learn** (for ML model)
- **Pandas** (for data handling)
- **Joblib** (for model saving/loading)

---

## 🧾 Project Files

| File Name | Description |
|------------|-------------|
| `app.py` | Streamlit app for classifying SMS messages |
| `train_balanced_model.py` | Script used to train the model |
| `sms_spam_model.joblib` | Saved trained model file |
| `sms+spam+collection.zip` | Dataset used for training |
| `sms_spam_training.ipynb` | Jupyter/Colab notebook for preprocessing and training |
| `requirements.txt` | Dependencies needed to run the project |

---

## 🚀 How to Run Locally

1. **Clone the Repository**
   ```bash
   git clone https://github.com/himadri999/SMS-Spam-Ham-Classifier.git
   cd SMS-Spam-Ham-Classifier

Install Required Package:

pip install -r requirements.txt

Run the Streamlit App:

streamlit run app.py


