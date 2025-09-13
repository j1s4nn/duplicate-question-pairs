# duplicate-question-pairs
A machine learning project to detect duplicate question pairs using NLP techniques such as text preprocessing, feature engineering, and similarity measures. The goal is to build a classifier that identifies whether two given questions have the same intent.

# Duplicate Question Pairs 🔍

A machine learning project to detect **duplicate question pairs** using NLP techniques.  
The aim is to identify whether two questions have the same intent or meaning.  

---

## 📌 Project Overview
- Task: Determine if two questions are semantically similar (duplicate or not).  
- Approach: Use **traditional ML techniques** instead of deep learning.  
- Applications: Search engines, Q&A forums (e.g., Quora, StackOverflow), chatbots.  

---

## 🗂️ Dataset
- Source: [Quora Question Pairs Dataset](https://www.kaggle.com/competitions/quora-question-pairs)  
- Contains pairs of questions with a label (`1 = duplicate`, `0 = not duplicate`).  

---

## ⚙️ Approach
1. **Data Preprocessing**
   - Tokenization, stopword removal, stemming/lemmatization.  
   - Lowercasing, punctuation removal.  

2. **Feature Engineering**
   - Bag-of-Words / TF-IDF.  
   - N-grams (bigrams, trigrams).  
   - Similarity scores (Cosine similarity, Jaccard similarity, Word overlap).  

3. **Modeling**
   - Logistic Regression.  
   - Support Vector Machine (SVM).  
   - Random Forest / Gradient Boosting.  

4. **Evaluation**
   - Accuracy, F1-score, Precision, Recall.  
   - Confusion Matrix.  

---

## 📊 Results
- Baseline: Random guess = ~50% accuracy.  
- Target: Achieve **70–80% accuracy** using ML techniques.  

---

## 🚀 Future Work
- Use **Word Embeddings** (Word2Vec, GloVe).  
- Extend to **Deep Learning models** (LSTM, BERT).  
- Deploy as a small **web app** for real-time duplicate detection.  

---

## 📫 Contact
👤 **MD Jisan Hossen**  
🎓 AI Undergraduate at Nanjing University of Information Science and Technology (NUIST)  
📍 Nanjing, Jiangsu Province, China  
🔗 [GitHub Profile](https://github.com/yourusername) | [LinkedIn](https://linkedin.com/in/yourusername)  

