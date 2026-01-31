# 🛡️ ScamShield – SCAM/HAM Detection System

ScamShield is an end-to-end **machine learning–based web application** designed to detect whether a text message or email is **SCAM or HAM**.

The project focuses on real-world **Data Science work beyond just model training**, including feature engineering, API development, databases, and deployment logic.

This project was built with the goal of **learning and applying practical Data Science concepts**, rather than creating a highly unique application.

---

## 🚀 Key Features

- SCAM vs HAM classification using Machine Learning  
- **TF-IDF vectorization** for text preprocessing  
- **Handcrafted linguistic features**  
- Feature fusion using **hstack**  
- Multiple models:
  - Logistic Regression
  - Neural Network  
- Configurable **probability threshold**  
- Flask-based web interface  
- Scan history stored in a database  
- **Unsupervised clustering** for message pattern analysis  
- Real application outputs and screenshots  

---

## 🧠 Machine Learning Pipeline

1. **Text Preprocessing**
   - Cleaning and normalization
   - TF-IDF vectorization

2. **Feature Engineering**
   - Handcrafted features (URLs, numbers, urgency words, etc.)
   - Combining sparse and dense features using `hstack`

3. **Model Training**
   - Logistic Regression (baseline and interpretable)
   - Neural Network (captures non-linear patterns)

4. **Prediction Logic**
   - Probability-based predictions
   - User-defined threshold for SCAM/HAM decision

5. **Unsupervised Clustering**
   - Groups similar messages based on content
   - Helps identify patterns such as:
     - Money-related scams
     - Employment or job-offer messages
     - Account verification scams
     - Data and privacy-related messages

---

## 🌐 Web Application

The web application allows users to:
- Paste a message or email
- Choose a model (Logistic Regression / Neural Network)
- Set a probability threshold
- Instantly get:
  - SCAM/HAM prediction
  - Risk score
- View scan history
- Explore clustering-based insights


## 🛠️ Tech Stack

- Python
- Scikit-learn
- TensorFlow / Keras
- Flask
- SQL / SQLite
- NumPy
- Pandas
- HTML / CSS


## 📂 Project Structure
```

ScamShield/
│
├── ml/ # Model training and prediction logic
├── web/ # Flask app, templates, static files
├── data/ # Datasets
├── database/ # Database-related files
├── outputs/ # Screenshots of application outputs
├── requirements.txt # Project dependencies
├── testing.py # Testing and experiments
└── .gitignore

```

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
python web/app.py
```
Then open your browser and got to: 
```
http://127.0.0.1:5000
```
## 📸 Application Outputs

Screenshots of the running application, model predictions, scan history, and clustering insights are available in the **`outputs/folder`**.

---

## 🎯 Learning Outcomes

Working on ScamShield helped me gain hands-on experience with:

- Feature Engineering  
- Machine Learning & Neural Networks  
- Probability-based decision systems  
- SQL and database integration  
- API and backend development  
- Web application development  
- Real-world Data Science workflows  

---

## 🙌 Acknowledgements

This project was built to understand the **realistic day-to-day work of a Data Scientist**, beyond just AI and Deep Learning models.

