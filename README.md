# LANGUAGE-DETECTION-MODEL

This repository contains a **Language Detection Model** that utilizes Natural Language Processing (NLP) techniques to predict the language of a given text. The model incorporates improved preprocessing, feature extraction, and training techniques to achieve accurate and robust predictions.

## Features

- **Data Preprocessing**: Tokenization, stopword removal, and lemmatization using `NLTK`.
- **Feature Engineering**: TF-IDF vectorization for efficient text representation.
- **Classification Model**: Logistic Regression for multi-class language detection.
- **Evaluation**: Detailed performance analysis with metrics and visualizations.
- **User Input**: Real-time prediction for user-provided text.

## Dataset

The model works with a multilingual dataset that includes a variety of languages. The dataset is loaded and preprocessed to handle missing values and ensure balanced representation across classes.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/language-detection-model.git
   cd language-detection-model
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure that `NLTK` resources are downloaded:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('wordnet')
   ```

## Usage

### 1. Train the Model
Run the notebook `IMPROVED_LANGUAGE_DETECTION_MODEL.ipynb` to preprocess the data, train the model, and evaluate its performance.

### 2. Predict Language
Use the interactive prediction cell in the notebook or integrate the model into your application:
```python
user_input = "Enter your text here"
user_tfidf = tfidf.transform([user_input])
prediction = model.predict(user_tfidf)
print(f"Predicted Language: {prediction[0]}")
```

## Results

- **Performance Metrics**: Includes classification accuracy, precision, recall, and F1-score.
- **Confusion Matrix**: Visual representation of model predictions.
- **Sample Visualization**: Language distribution bar chart and other exploratory data analysis plots.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

