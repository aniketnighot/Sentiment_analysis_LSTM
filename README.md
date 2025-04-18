# 🎬 Sentiment Analysis on IMDB Reviews

This project performs sentiment analysis on movie reviews from the IMDB dataset using an LSTM-based model built with PyTorch. The app also includes a frontend interface to test custom reviews!

## 🧠 Technologies Used

- Python, PyTorch, NumPy, Pandas
- LSTM for NLP classification
- Frontend built using Flask
- Dataset: IMDB reviews

## 💡 Features

- Train and evaluate an LSTM model
- Predict sentiment of custom text
- Web UI for interactive testing
- Cleaned and preprocessed text data

## 📌 Key Steps

### 1. 🧪 Data Exploration
- The dataset contains **50,000 reviews**, evenly split between positive and negative sentiments.
- Review lengths vary significantly:
  - **Average:** 231 words  
  - **Maximum:** 2470 words  
  - **Minimum:** 4 words

---

### 2. 🧼 Data Preprocessing
The preprocessing pipeline includes:

- Converting text to lowercase  
- Removing HTML tags  
- Cleaning special characters and URLs  
- Tokenizing text into individual words  
- Removing common stopwords  
- Stemming words to their root form

---

### 3. 🧠 Model Architecture
An LSTM-based sentiment analysis model was implemented with:

- **Embedding layer:** 128 dimensions  
- **Dropout layer:** 0.3  
- **LSTM layer:** 128 hidden units, 2 layers  
- **Linear output layer:** 1 unit (binary classification)

---

### 4. 🏋️ Training Process
- Trained for **13 epochs** (early stopping triggered)  
- **Training Accuracy:** 95.04%  
- **Validation Accuracy:** 86.57%  
- **Best Validation Loss:** 0.3294

---

### 5. 📈 Results
The model shows good performance on both training and validation sets:

- Steady decrease in training loss  
- Good generalization to validation data  
- Early stopping effectively prevented overfitting

---

## 🔮 Next Steps

- Further hyperparameter tuning to improve performance  
- Testing on the held-out test set for final evaluation  
- Exploring more advanced architectures like **BERT** for comparison

---

> 📝 The notebook demonstrates a complete workflow from data exploration to model training for sentiment analysis on IMDB reviews. The LSTM model shows promising results in classifying movie reviews as positive or negative.
