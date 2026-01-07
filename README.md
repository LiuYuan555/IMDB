# IMDB Sentiment Classification Project

A comprehensive machine learning project that applies multiple approaches to classify movie reviews from the IMDB dataset as positive or negative sentiment.

## Project Overview

This project explores various natural language processing (NLP) and machine learning techniques for sentiment analysis:
- **Logistic Regression**: Traditional machine learning approach
- **TF-IDF Vectorization**: Text feature extraction
- **Neural Networks**: Deep learning approach with TensorFlow/Keras
- **Pandas & Tensor Operations**: Data manipulation and exploration

## Dataset

**IMDB Dataset.csv** - Contains 50,000 movie reviews with sentiment labels
- 25,000 positive reviews
- 25,000 negative reviews
- Columns: `review`, `sentiment`

After deduplication, the dataset contains 49,582 unique reviews.

## Project Structure

```
imdb/
├── IMDB Dataset.csv                    # Main dataset
├── README.md                           # This file
├── imdb_neural_network.ipynb          # Main neural network model
├── logistic_regression.ipynb          # Baseline ML approach
├── tfidf_vectorizer_explained.ipynb   # TF-IDF feature extraction
├── tensor.ipynb                       # Tensor operations foundation
├── pandas_values_explanation.ipynb    # Data exploration with Pandas
└── test.ipynb                         # Testing and experiments
```

## Notebooks Description

### 1. **imdb_neural_network.ipynb** ⭐ Main Model
The primary notebook implementing a deep neural network for sentiment classification.

**Key Steps:**
- Load and prepare IMDB dataset
- Remove duplicates and analyze sentiment distribution
- Vectorize text using TF-IDF (5,000 features)
- Build a sequential neural network with:
  - Input layer: 5,000 features
  - Hidden layers: 128, 64, 32 neurons with ReLU activation
  - Dropout layers (0.2-0.3) to prevent overfitting
  - Output layer: 1 neuron with sigmoid activation for binary classification
- Train with early stopping (patience=2 epochs)
- Evaluate on test set and generate confusion matrix
- Make predictions on custom reviews

**Performance:**
- Test Accuracy: **88.13%**
- Test Loss: 0.423

**Sample Predictions:**
- Positive reviews: Confidence ~98%
- Negative reviews: Confidence ~99.9%

### 2. **logistic_regression.ipynb** 📊 Baseline Model
Traditional machine learning approach for comparison.

### 3. **tfidf_vectorizer_explained.ipynb** 🔤 Feature Engineering
Detailed explanation of TF-IDF vectorization for converting text to numerical features.

### 4. **tensor.ipynb** 📐 Foundations
Tensor operations and neural network basics (foundation for imdb_neural_network.ipynb).

### 5. **pandas_values_explanation.ipynb** 📈 Data Exploration
Data manipulation and exploration using Pandas.

### 6. **test.ipynb** 🧪 Testing
Experimental cells and additional tests.

## Installation & Setup

### Prerequisites
- Python 3.7+
- Jupyter Notebook

### Required Libraries
```bash
pip install tensorflow keras scikit-learn pandas numpy matplotlib seaborn
```

Or install from requirements (if available):
```bash
pip install -r requirements.txt
```

## Usage

### Running the Main Model

1. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Open `imdb_neural_network.ipynb`** and run cells sequentially

3. **Key Cells:**
   - Cells 1-3: Import libraries and load data
   - Cells 4-8: Data preparation and vectorization
   - Cell 9: Build neural network model
   - Cell 10: Train the model with early stopping
   - Cell 11: Plot training history
   - Cell 12: Evaluate on test set
   - Cells 13-16: Generate confusion matrix and classification report
   - Cell 17: Test on custom reviews

### Testing on Custom Reviews

Modify the `custom_reviews` list in the final cell:

```python
custom_reviews = [
    'This movie was absolutely amazing! I loved every minute of it.',
    'Terrible waste of time. Worst movie I have ever seen.',
    'It was okay, not great but not bad either.'
]
```

The model will predict sentiment as **POSITIVE** or **NEGATIVE** with confidence scores.

## Model Architecture

```
Input Layer (5000 features from TF-IDF)
    ↓
Dense(128) + ReLU + Dropout(0.3)
    ↓
Dense(64) + ReLU + Dropout(0.3)
    ↓
Dense(32) + ReLU + Dropout(0.2)
    ↓
Dense(1) + Sigmoid (Binary Classification Output)
```

**Regularization Techniques:**
- L2 regularization (kernel_regularizer=0.001)
- Dropout layers (20-30%)
- Early stopping (monitor validation loss)

## Results & Performance

### Confusion Matrix
```
                Predicted
              Negative | Positive
Actual  Negative  4241   |   699
        Positive   478   | 4499
```

### Classification Metrics
- **Accuracy**: 88.13%
- **Precision (Positive)**: ~86.5%
- **Recall (Positive)**: ~90.4%
- **F1-Score (Positive)**: ~88.4%

### Training Behavior
- Early stopping typically activates around epoch 4-5
- Validation loss stabilizes after epoch 3
- No significant overfitting due to dropout and regularization

## Key Features

✅ **Data Preprocessing**
- Duplicate removal
- Sentiment encoding (0/1)
- Train-test split (80/20)

✅ **Text Vectorization**
- TF-IDF with 5,000 max features
- English stop words removed
- Sparse to dense matrix conversion

✅ **Neural Network**
- Multi-layer architecture
- Dropout for regularization
- L2 weight regularization
- Early stopping callback

✅ **Comprehensive Evaluation**
- Accuracy, precision, recall, F1-score
- Confusion matrix visualization
- Custom review testing

## Customization Options

### Adjust TF-IDF Vectorizer
```python
vectorizer = TfidfVectorizer(
    max_features=10000,  # Increase/decrease features
    stop_words='english',
    ngram_range=(1, 2)   # Include bigrams
)
```

### Modify Neural Network
```python
model = keras.Sequential([
    keras.layers.Dense(256, activation='relu', ...),  # Larger layers
    keras.layers.Dropout(0.4),  # More dropout
    # Add more layers as needed
])
```

### Training Parameters
```python
history = model.fit(
    X_train_vectorized, y_train,
    epochs=20,          # More epochs
    batch_size=64,      # Larger batches
    validation_split=0.25,  # More validation data
    callbacks=[early_stopping]
)
```

## Troubleshooting

**Issue**: Out of memory when loading dataset
- **Solution**: Reduce `max_features` in TfidfVectorizer

**Issue**: Model overfitting (large train/val loss gap)
- **Solution**: Increase dropout rates or L2 regularization

**Issue**: Poor predictions on custom reviews
- **Solution**: Ensure text is preprocessed similarly (lowercase, punctuation handling)

## Future Improvements

- [ ] Add word embeddings (Word2Vec, GloVe)
- [ ] Implement attention mechanisms
- [ ] Use pre-trained BERT/RoBERTa models
- [ ] Add data augmentation techniques
- [ ] Implement cross-validation for better evaluation
- [ ] Deploy as REST API using Flask/FastAPI

## References

- TensorFlow/Keras Documentation: https://www.tensorflow.org/
- Scikit-learn TF-IDF: https://scikit-learn.org/
- IMDB Dataset: http://ai.stanford.edu/~amaas/data/sentiment/

## Author

Created for exploring NLP and deep learning techniques in sentiment analysis.

## License

This project is provided for educational purposes.
