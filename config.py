import json
import os

class Config:
    """Configuration file for sentiment analysis model"""
    
    # Model parameters
    VOCAB_SIZE = 5000
    MAX_LENGTH = 150
    EMBEDDING_DIM = 100
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 10
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.2
    
    # LSTM parameters
    LSTM_UNITS = 100
    DROPOUT_RATE = 0.2
    RECURRENT_DROPOUT = 0.2
    DENSE_UNITS_1 = 64
    DENSE_UNITS_2 = 32
    DROPOUT_DENSE = 0.5
    
    # Training settings
    LEARNING_RATE = 0.001
    OPTIMIZER = 'adam'
    LOSS_FUNCTION = 'binary_crossentropy'
    METRICS = ['accuracy']
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_MONITOR = 'val_loss'
    
    # Learning rate reduction
    REDUCE_LR_FACTOR = 0.5
    REDUCE_LR_PATIENCE = 2
    MIN_LEARNING_RATE = 0.00001
    
    # Data paths
    DATA_PATH = 'sentiment_data.csv'
    MODEL_PATH = 'sentiment_model.h5'
    TOKENIZER_PATH = 'tokenizer.pkl'
    
    # NLP settings
    LANGUAGE = 'english'
    REMOVE_STOPWORDS = True
    LOWERCASE = True
    REMOVE_SPECIAL_CHARS = True
    
    @classmethod
    def to_dict(cls):
        """Convert config to dictionary"""
        return {k: v for k, v in cls.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def save_config(cls, filepath='config.json'):
        """Save configuration to JSON file"""
        config_dict = cls.to_dict()
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)
        print(f"Configuration saved to {filepath}")
    
    @classmethod
    def load_config(cls, filepath='config.json'):
        """Load configuration from JSON file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            
            # Update class attributes
            for key, value in config_dict.items():
                setattr(cls, key, value)
            print(f"Configuration loaded from {filepath}")
        else:
            print(f"Config file {filepath} not found. Using default configuration.")


if __name__ == '__main__':
    # Display configuration
    print("=" * 50)
    print("SENTIMENT ANALYSIS MODEL CONFIGURATION")
    print("=" * 50)
    
    config_dict = Config.to_dict()
    
    # Group configurations by category
    categories = {
        'Model Parameters': ['VOCAB_SIZE', 'MAX_LENGTH', 'EMBEDDING_DIM'],
        'Training Parameters': ['BATCH_SIZE', 'EPOCHS', 'VALIDATION_SPLIT', 'TEST_SPLIT'],
        'LSTM Architecture': ['LSTM_UNITS', 'DROPOUT_RATE', 'RECURRENT_DROPOUT', 
                             'DENSE_UNITS_1', 'DENSE_UNITS_2', 'DROPOUT_DENSE'],
        'Optimization': ['LEARNING_RATE', 'OPTIMIZER', 'LOSS_FUNCTION', 'METRICS'],
        'Early Stopping': ['EARLY_STOPPING_PATIENCE', 'EARLY_STOPPING_MONITOR'],
        'Learning Rate': ['REDUCE_LR_FACTOR', 'REDUCE_LR_PATIENCE', 'MIN_LEARNING_RATE'],
        'Data Paths': ['DATA_PATH', 'MODEL_PATH', 'TOKENIZER_PATH'],
        'NLP Settings': ['LANGUAGE', 'REMOVE_STOPWORDS', 'LOWERCASE', 'REMOVE_SPECIAL_CHARS']
    }
    
    for category, keys in categories.items():
        print(f"\n{category}:")
        print("-" * 50)
        for key in keys:
            if key in config_dict:
                print(f"  {key}: {config_dict[key]}")
    
    print("\n" + "=" * 50)
    
    # Save configuration
    Config.save_config('config.json')
