import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

class SentimentAnalysisLSTM:
    def __init__(self, vocab_size=5000, max_length=150, embedding_dim=100):
        """
        Initialize the Sentiment Analysis LSTM model
        
        Args:
            vocab_size: Maximum number of words to keep in vocabulary
            max_length: Maximum length of sequences
            embedding_dim: Dimension of word embeddings
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.label_encoder = None
        self.model = None
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)
    
    def prepare_data(self, df):
        """Prepare data for model training"""
        # Preprocess texts
        df['cleaned_text'] = df['review'].apply(self.preprocess_text)
        
        # Tokenize
        self.tokenizer = Tokenizer(num_words=self.vocab_size)
        self.tokenizer.fit_on_texts(df['cleaned_text'])
        sequences = self.tokenizer.texts_to_sequences(df['cleaned_text'])
        
        # Pad sequences
        X = pad_sequences(sequences, maxlen=self.max_length)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['sentiment'])
        
        return X, y
    
    def build_model(self):
        """Build the LSTM model architecture"""
        self.model = Sequential([
            Embedding(input_dim=self.vocab_size, 
                     output_dim=self.embedding_dim, 
                     input_length=self.max_length),
            SpatialDropout1D(0.2),
            LSTM(100, dropout=0.2, recurrent_dropout=0.2),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        """Train the model"""
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict_sentiment(self, text):
        """Predict sentiment for a single text"""
        cleaned = self.preprocess_text(text)
        sequence = self.tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=self.max_length)
        
        prediction = self.model.predict(padded, verbose=0)[0][0]
        sentiment = self.label_encoder.inverse_transform([1 if prediction > 0.5 else 0])[0]
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        return sentiment, confidence
    
    def save_model(self, model_path='sentiment_model.h5', tokenizer_path='tokenizer.pkl'):
        """Save model and tokenizer"""
        self.model.save(model_path)
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print(f"Model saved to {model_path}")
        print(f"Tokenizer saved to {tokenizer_path}")
    
    def load_model(self, model_path='sentiment_model.h5', tokenizer_path='tokenizer.pkl'):
        """Load saved model and tokenizer"""
        from tensorflow.keras.models import load_model
        self.model = load_model(model_path)
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        print(f"Model loaded from {model_path}")
        print(f"Tokenizer loaded from {tokenizer_path}")


if __name__ == '__main__':
    # Load data
    df = pd.read_csv('sentiment_data.csv')
    
    print("Dataset shape:", df.shape)
    print("\nSentiment distribution:")
    print(df['sentiment'].value_counts())
    
    # Initialize model
    sentiment_model = SentimentAnalysisLSTM(vocab_size=5000, max_length=150, embedding_dim=100)
    
    # Prepare data
    X, y = sentiment_model.prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Validation set size: {X_val.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Build and train model
    sentiment_model.build_model()
    print("\nModel Architecture:")
    sentiment_model.model.summary()
    
    print("\nTraining model...")
    history = sentiment_model.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
    
    # Evaluate on test set
    test_loss, test_accuracy = sentiment_model.model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save model
    sentiment_model.save_model()
    
    # Example predictions
    print("\n--- Example Predictions ---")
    test_reviews = [
        "This product is absolutely amazing! I love it.",
        "Terrible quality. Very disappointed with my purchase.",
        "It's okay, nothing special but does the job."
    ]
    
    for review in test_reviews:
        sentiment, confidence = sentiment_model.predict_sentiment(review)
        print(f"Review: {review}")
        print(f"Sentiment: {sentiment} (Confidence: {confidence:.2%})\n")
