import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')


class ModelUtils:
    """Utility functions for sentiment analysis model"""
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        return cm
    
    @staticmethod
    def print_classification_report(y_true, y_pred):
        """Print detailed classification report"""
        report = classification_report(y_true, y_pred, 
                                      target_names=['Negative', 'Positive'])
        print("\n" + "=" * 50)
        print("CLASSIFICATION REPORT")
        print("=" * 50)
        print(report)
    
    @staticmethod
    def plot_roc_curve(y_true, y_pred_proba):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
        
        return roc_auc
    
    @staticmethod
    def plot_training_history(history, save_path=None):
        """Plot training and validation metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
        axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate (if available)
        if 'lr' in history.history:
            axes[1, 0].plot(history.history['lr'], label='Learning Rate', linewidth=2)
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')
        
        # Summary statistics
        axes[1, 1].axis('off')
        summary_text = f"""
        TRAINING SUMMARY
        {'=' * 30}
        Total Epochs: {len(history.history['loss'])}
        Final Train Accuracy: {history.history['accuracy'][-1]:.4f}
        Final Val Accuracy: {history.history['val_accuracy'][-1]:.4f}
        Final Train Loss: {history.history['loss'][-1]:.4f}
        Final Val Loss: {history.history['val_loss'][-1]:.4f}
        Best Val Accuracy: {max(history.history['val_accuracy']):.4f}
        Best Val Loss: {min(history.history['val_loss']):.4f}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, 
                       family='monospace', verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def analyze_misclassified(texts, y_true, y_pred, label_encoder):
        """Analyze misclassified samples"""
        misclassified_indices = np.where(y_true != y_pred)[0]
        
        print("\n" + "=" * 70)
        print(f"MISCLASSIFIED SAMPLES ({len(misclassified_indices)} total)")
        print("=" * 70)
        
        for idx, i in enumerate(misclassified_indices[:10]):  # Show first 10
            true_label = label_encoder.inverse_transform([y_true[i]])[0]
            pred_label = label_encoder.inverse_transform([y_pred[i]])[0]
            
            print(f"\n{idx + 1}. Review:")
            print(f"   Text: {texts.iloc[i][:100]}...")
            print(f"   True: {true_label}")
            print(f"   Predicted: {pred_label}")
        
        print("\n" + "=" * 70)
    
    @staticmethod
    def get_model_summary(model):
        """Get detailed model summary"""
        print("\n" + "=" * 70)
        print("MODEL ARCHITECTURE")
        print("=" * 70)
        model.summary()
        
        print("\n" + "=" * 70)
        print("MODEL CONFIGURATION")
        print("=" * 70)
        print(f"Optimizer: {model.optimizer.__class__.__name__}")
        print(f"Loss Function: {model.loss}")
        print(f"Metrics: {model.metrics}")
        print(f"Total Parameters: {model.count_params():,}")
        print("=" * 70)
    
    @staticmethod
    def create_sample_batch(tokenizer, texts, max_length):
        """Create a batch of padded sequences from texts"""
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=max_length)
        return padded
    
    @staticmethod
    def word_frequency_analysis(df, tokenizer, top_n=20):
        """Analyze word frequency in positive and negative reviews"""
        from collections import Counter
        
        positive_reviews = df[df['sentiment'] == 'positive']['review'].str.lower()
        negative_reviews = df[df['sentiment'] == 'negative']['review'].str.lower()
        
        # Tokenize and count
        positive_tokens = []
        negative_tokens = []
        
        for review in positive_reviews:
            tokens = review.split()
            positive_tokens.extend(tokens)
        
        for review in negative_reviews:
            tokens = review.split()
            negative_tokens.extend(tokens)
        
        positive_freq = Counter(positive_tokens).most_common(top_n)
        negative_freq = Counter(negative_tokens).most_common(top_n)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        pos_words, pos_counts = zip(*positive_freq)
        axes[0].barh(range(len(pos_words)), pos_counts, color='green', alpha=0.7)
        axes[0].set_yticks(range(len(pos_words)))
        axes[0].set_yticklabels(pos_words)
        axes[0].invert_yaxis()
        axes[0].set_title(f'Top {top_n} Words in Positive Reviews')
        axes[0].set_xlabel('Frequency')
        
        neg_words, neg_counts = zip(*negative_freq)
        axes[1].barh(range(len(neg_words)), neg_counts, color='red', alpha=0.7)
        axes[1].set_yticks(range(len(neg_words)))
        axes[1].set_yticklabels(neg_words)
        axes[1].invert_yaxis()
        axes[1].set_title(f'Top {top_n} Words in Negative Reviews')
        axes[1].set_xlabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
        return positive_freq, negative_freq


if __name__ == '__main__':
    print("Model utility functions loaded successfully!")
    print("\nAvailable functions:")
    print("- plot_confusion_matrix()")
    print("- print_classification_report()")
    print("- plot_roc_curve()")
    print("- plot_training_history()")
    print("- analyze_misclassified()")
    print("- get_model_summary()")
    print("- create_sample_batch()")
    print("- word_frequency_analysis()")
