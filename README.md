# Sentiment Analysis with LSTM and NLP

A comprehensive Python project for sentiment analysis using Long Short-Term Memory (LSTM) neural networks combined with Natural Language Processing (NLP) techniques.

## Project Overview

This project implements a deep learning model to classify text reviews as positive or negative sentiments. It demonstrates the application of:
- **LSTM Networks**: For sequential text processing
- **Word Embeddings**: For semantic representation of words
- **NLP Preprocessing**: Including tokenization, stopword removal, and text cleaning

## Features

- **Text Preprocessing**: Automatic cleaning, tokenization, and stopword removal
- **LSTM Architecture**: Multi-layer LSTM with dropout and regularization
- **Model Evaluation**: Comprehensive metrics and visualization
- **Sentiment Prediction**: Easy-to-use interface for predicting sentiment on new text
- **Model Persistence**: Save and load trained models

## Project Structure

```
sentimentanalysis/
├── sentiment_analysis.py           # Main model implementation
├── sentiment_analysis_notebook.ipynb # Interactive Jupyter notebook
├── sentiment_data.csv              # Sample training dataset
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Mayilarasi/sentimentanalysis.git
cd sentimentanalysis
```

2. Create a virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the main script to train the model on the provided dataset:

```bash
python sentiment_analysis.py
```

This will:
- Load the sentiment dataset
- Preprocess and tokenize the text
- Train the LSTM model
- Evaluate on test set
- Save the model for future use
- Generate example predictions

### Using the Model for Predictions

```python
from sentiment_analysis import SentimentAnalysisLSTM

# Initialize and load model
model = SentimentAnalysisLSTM()
model.load_model('sentiment_model.h5', 'tokenizer.pkl')

# Predict sentiment
review = "This product is absolutely amazing!"
sentiment, confidence = model.predict_sentiment(review)

print(f"Sentiment: {sentiment}")
print(f"Confidence: {confidence:.2%}")
```

### Interactive Analysis

Use the Jupyter notebook for interactive exploration:

```bash
jupyter notebook sentiment_analysis_notebook.ipynb
```

## Model Architecture

```
Input (Sequence of word indices)
    ↓
Embedding Layer (Vocab Size: 5000, Embedding Dim: 100)
    ↓
Spatial Dropout (0.2)
    ↓
LSTM Layer (100 units, Dropout: 0.2, Recurrent Dropout: 0.2)
    ↓
Dense Layer (64 units, ReLU)
    ↓
Dropout (0.5)
    ↓
Dense Layer (32 units, ReLU)
    ↓
Dropout (0.5)
    ↓
Output Layer (1 unit, Sigmoid)
    ↓
Binary Classification (Positive/Negative)
```

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Vocab Size | 5000 | Maximum vocabulary size |
| Max Length | 150 | Maximum sequence length |
| Embedding Dim | 100 | Word embedding dimension |
| LSTM Units | 100 | LSTM layer units |
| Batch Size | 32 | Training batch size |
| Epochs | 10 | Maximum training epochs |
| Optimizer | Adam | Learning rate: 0.001 |
| Loss Function | Binary Crossentropy | For binary classification |

## Dataset

The `sentiment_data.csv` file contains:
- **review**: Text content of the review
- **sentiment**: Label (positive/negative)

Sample data is included for demonstration. For better results, use a larger dataset like:
- [IMDB Movie Reviews](http://ai.stanford.edu/~amaas/data/sentiment/)
- [Amazon Product Reviews](https://github.com/nizarjomaa/Amazon-reviews)
- [Yelp Reviews](https://www.yelp.com/dataset)

## Performance Metrics

The model evaluates performance using:
- **Accuracy**: Overall correct predictions
- **Loss**: Binary crossentropy loss
- **Confidence**: Probability scores for predictions

## Requirements

- Python 3.8+
- TensorFlow 2.12+
- NumPy, Pandas
- NLTK for NLP preprocessing
- Scikit-learn for data splitting
- Matplotlib, Seaborn for visualization

See `requirements.txt` for complete dependencies.

## Future Enhancements

- [ ] Implement Bidirectional LSTM (BiLSTM)
- [ ] Add attention mechanisms
- [ ] Support for multi-class sentiment (positive, negative, neutral)
- [ ] Integration with pre-trained embeddings (Word2Vec, GloVe)
- [ ] Web API for sentiment prediction
- [ ] Real-time sentiment analysis from social media

## Key Concepts

### LSTM (Long Short-Term Memory)
LSTM is a type of recurrent neural network (RNN) that can learn long-term dependencies in sequential data. It's particularly effective for sentiment analysis as it captures contextual information in text.

### NLP Preprocessing
Natural Language Processing techniques are applied to clean and normalize text:
- Lowercasing
- Removing special characters
- Tokenization
- Stopword removal

### Word Embeddings
Words are transformed into dense numerical vectors that capture semantic meaning, allowing the model to understand relationships between words.

## License

This project is open source and available under the MIT License.

## Author

Sentiment Analysis LSTM Model
- Created: 2026
- Repository: https://github.com/Mayilarasi/sentimentanalysis

## References

- [LSTM Networks for Sentiment Analysis](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Natural Language Processing with Deep Learning](https://cs224d.stanford.edu/)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/)
- [NLTK Documentation](https://www.nltk.org/)

## Support

For issues, questions, or suggestions, please create an issue on GitHub.
