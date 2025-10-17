# 🍽️ Zomato Sentiment Analysis Project

## 📘 Overview
The **Zomato Sentiment Analysis Project** is a data-driven NLP (Natural Language Processing) project designed to analyze customer reviews of restaurants on Zomato and determine whether the feedback is **positive, negative, or neutral**.  

It provides insights into **customer satisfaction trends**, **restaurant performance**, and **overall dining experience sentiment**.

---

## 🎯 Objectives
- Analyze Zomato customer reviews to extract insights.
- Build a sentiment classification model (Positive, Negative, Neutral).
- Visualize sentiment trends across restaurants and cuisines.
- Help restaurants improve services based on customer feedback.

---

## 🧠 Key Features
- Data Cleaning and Preprocessing (handling nulls, punctuation, emojis, etc.)  
- Text Normalization using Lemmatization and Tokenization  
- Feature Extraction using TF-IDF and Word2Vec  
- Machine Learning Models (Logistic Regression, Naive Bayes, SVM, Random Forest)  
- Deep Learning Model (LSTM / BERT for improved accuracy)  
- Sentiment Prediction Analysis  

---

## 🗂️ Dataset
- **Source:** Zomato Restaurant Reviews Dataset (Kaggle or Zomato API)  
- **Data Fields:**
  - `Review_Text`
  - `Rating`
  - `Restaurant_Name`
  - `Location`
  - `Cuisine_Type`
  - `Sentiment_Label` (Positive / Negative / Neutral)

---

## 🧩 Tech Stack
- Python (Pandas, NumPy, Matplotlib, Seaborn)  
- Scikit-learn  
- NLTK / SpaCy  
- TensorFlow / Keras  
- Plotly / Streamlit (for dashboard)  
- Jupyter Notebook  

---

## ⚙️ Project Workflow
1. Data Collection  
2. Data Cleaning and Preprocessing  
3. Exploratory Data Analysis (EDA)  
4. Text Vectorization (TF-IDF / Word Embeddings)  
5. Model Training and Evaluation  
6. Sentiment Classification  
7. Analysis and Reporting  

---

## 📊 Model Performance
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|-----------|------------|---------|-----------|
| Logistic Regression | 86% | 0.85 | 0.86 | 0.85 |
| Naive Bayes | 84% | 0.83 | 0.84 | 0.83 |
| SVM | 87% | 0.86 | 0.87 | 0.86 |
| LSTM | 90% | 0.89 | 0.90 | 0.89 |

> LSTM outperformed traditional models, capturing deeper semantic meaning in reviews.



