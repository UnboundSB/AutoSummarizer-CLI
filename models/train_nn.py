import pandas as pd
import numpy as np
import spacy
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input, Attention, TimeDistributed
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import re
from collections import Counter
import warnings

def create_and_train_pos_lstm_summarizer(csv_file_path, 
                                       document_column='document', 
                                       summary_column='summary',
                                       max_doc_length=500,
                                       max_summary_length=100,
                                       embedding_dim=128,
                                       lstm_units=256,
                                       epochs=50,
                                       batch_size=32,
                                       test_size=0.2,
                                       model_save_path='summarizer_model.h5',
                                       encoders_save_path='encoders.pkl',
                                       min_doc_length=5,
                                       min_summary_length=2,
                                       vocab_min_freq=2,
                                       handle_empty_strings=True,
                                       auto_adjust_lengths=True):
    """
    Complete function to create and train a POS-based LSTM summarizer model with robust edge case handling.
    
    Parameters:
    - csv_file_path: Path to CSV file with document and summary columns
    - document_column: Name of document column in CSV
    - summary_column: Name of summary column in CSV
    - max_doc_length: Maximum length for input documents (in tokens)
    - max_summary_length: Maximum length for output summaries (in tokens)
    - embedding_dim: Dimension of word embeddings
    - lstm_units: Number of LSTM units
    - epochs: Training epochs
    - batch_size: Training batch size
    - test_size: Fraction of data for testing
    - model_save_path: Path to save trained model
    - encoders_save_path: Path to save encoders and tokenizers
    - min_doc_length: Minimum document length (shorter docs will be padded/skipped)
    - min_summary_length: Minimum summary length (shorter summaries will be padded/skipped)
    - vocab_min_freq: Minimum frequency for word to be included in vocabulary
    - handle_empty_strings: Whether to handle empty/null strings
    - auto_adjust_lengths: Whether to automatically adjust max lengths based on data
    
    Returns:
    - trained_model: Trained Keras model
    - history: Training history
    - encoders: Dictionary with encoders and tokenizers
    """
    
    print("Loading spaCy model...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Please install spaCy English model: python -m spacy download en_core_web_sm")
        return None, None, None
    
    print("Loading and preprocessing data...")
    
    # Load CSV data with error handling
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None, None, None
    
    # Check if columns exist
    if document_column not in df.columns:
        print(f"Column '{document_column}' not found in CSV")
        return None, None, None
    
    if summary_column not in df.columns:
        print(f"Column '{summary_column}' not found in CSV")
        return None, None, None
    
    # Handle missing/null values
    print("Handling missing values...")
    original_size = len(df)
    df = df.dropna(subset=[document_column, summary_column])
    
    if handle_empty_strings:
        # Remove empty strings and whitespace-only strings
        df = df[df[document_column].astype(str).str.strip() != '']
        df = df[df[summary_column].astype(str).str.strip() != '']
    
    print(f"Removed {original_size - len(df)} rows with missing/empty data")
    
    if len(df) == 0:
        print("No valid data remaining after cleaning!")
        return None, None, None
    
    documents = df[document_column].astype(str).tolist()
    summaries = df[summary_column].astype(str).tolist()
    
    print(f"Loaded {len(documents)} valid document-summary pairs")
    
    # Clean text function with enhanced handling
    def clean_text(text):
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text).lower()
        # Remove excessive whitespace and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    # Enhanced processing with POS tagging and edge case handling
    def process_with_pos(texts, max_length, text_type="document"):
        processed_texts = []
        pos_sequences = []
        valid_indices = []
        
        min_length = min_doc_length if text_type == "document" else min_summary_length
        
        for idx, text in enumerate(texts):
            try:
                cleaned_text = clean_text(text)
                
                if len(cleaned_text.strip()) == 0:
                    print(f"Warning: Empty {text_type} at index {idx}, skipping")
                    continue
                
                doc = nlp(cleaned_text)
                tokens = []
                pos_tags = []
                
                # Extract tokens and POS tags with better filtering
                for token in doc:
                    if not token.is_space and not token.is_punct and len(token.text.strip()) > 0:
                        # Handle numeric tokens
                        if token.like_num:
                            tokens.append("<NUM>")
                        elif token.is_alpha and len(token.text) > 1:
                            tokens.append(token.text)
                        elif len(token.text.strip()) > 0:
                            tokens.append(token.text)
                        
                        pos_tags.append(token.pos_)
                
                # Handle very short sequences
                if len(tokens) < min_length:
                    if len(tokens) == 0:
                        print(f"Warning: No valid tokens in {text_type} at index {idx}, skipping")
                        continue
                    else:
                        # Pad very short sequences by repeating tokens
                        while len(tokens) < min_length:
                            tokens.extend(tokens[:min(len(tokens), min_length - len(tokens))])
                            pos_tags.extend(pos_tags[:min(len(pos_tags), min_length - len(pos_tags))])
                
                # Truncate if too long
                if len(tokens) > max_length:
                    tokens = tokens[:max_length]
                    pos_tags = pos_tags[:max_length]
                
                # Ensure pos_tags matches tokens length
                if len(pos_tags) != len(tokens):
                    pos_tags = pos_tags[:len(tokens)]
                    if len(pos_tags) < len(tokens):
                        pos_tags.extend(['NOUN'] * (len(tokens) - len(pos_tags)))
                
                processed_texts.append(tokens)
                pos_sequences.append(pos_tags)
                valid_indices.append(idx)
                
            except Exception as e:
                print(f"Error processing {text_type} at index {idx}: {e}")
                continue
        
        return processed_texts, pos_sequences, valid_indices
    
    print("Processing documents with POS tagging...")
    doc_tokens, doc_pos, valid_doc_indices = process_with_pos(documents, max_doc_length, "document")
    
    print("Processing summaries...")
    summary_tokens, _, valid_summary_indices = process_with_pos(summaries, max_summary_length, "summary")
    
    # Find intersection of valid indices
    valid_indices = list(set(valid_doc_indices) & set(valid_summary_indices))
    
    if len(valid_indices) == 0:
        print("No valid document-summary pairs after processing!")
        return None, None, None
    
    # Filter to only valid pairs
    doc_indices_map = {idx: i for i, idx in enumerate(valid_doc_indices)}
    summary_indices_map = {idx: i for i, idx in enumerate(valid_summary_indices)}
    
    final_doc_tokens = []
    final_doc_pos = []
    final_summary_tokens = []
    
    for idx in valid_indices:
        if idx in doc_indices_map and idx in summary_indices_map:
            final_doc_tokens.append(doc_tokens[doc_indices_map[idx]])
            final_doc_pos.append(doc_pos[doc_indices_map[idx]])
            final_summary_tokens.append(summary_tokens[summary_indices_map[idx]])
    
    doc_tokens, doc_pos, summary_tokens = final_doc_tokens, final_doc_pos, final_summary_tokens
    
    print(f"Final dataset size: {len(doc_tokens)} valid pairs")
    
    if len(doc_tokens) < 10:
        print("Warning: Very small dataset! Consider relaxing filtering criteria.")
    
    # Auto-adjust lengths based on data distribution if enabled
    if auto_adjust_lengths:
        doc_lengths = [len(tokens) for tokens in doc_tokens]
        summary_lengths = [len(tokens) for tokens in summary_tokens]
        
        # Use 95th percentile for max lengths to avoid extreme outliers
        recommended_doc_length = int(np.percentile(doc_lengths, 95))
        recommended_summary_length = int(np.percentile(summary_lengths, 95))
        
        if recommended_doc_length != max_doc_length:
            print(f"Recommended max_doc_length: {recommended_doc_length} (current: {max_doc_length})")
            max_doc_length = min(recommended_doc_length, max_doc_length * 2)  # Cap at 2x original
        
        if recommended_summary_length != max_summary_length:
            print(f"Recommended max_summary_length: {recommended_summary_length} (current: {max_summary_length})")
            max_summary_length = min(recommended_summary_length, max_summary_length * 2)  # Cap at 2x original
    
    # Create vocabularies with better handling
    print("Building vocabularies...")
    
    # Word vocabulary with frequency filtering
    all_words = []
    for tokens in doc_tokens + summary_tokens:
        all_words.extend(tokens)
    
    word_counts = Counter(all_words)
    print(f"Total unique words before filtering: {len(word_counts)}")
    
    # Keep words that appear at least vocab_min_freq times
    vocab_words = [word for word, count in word_counts.items() if count >= vocab_min_freq]
    
    # Add special tokens
    special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
    word_vocab = special_tokens + vocab_words
    
    # POS vocabulary with all possible tags
    all_pos = []
    for pos_seq in doc_pos:
        all_pos.extend(pos_seq)
    
    unique_pos = list(set(all_pos))
    pos_vocab = ['<PAD>'] + unique_pos
    
    print(f"Final word vocabulary size: {len(word_vocab)}")
    print(f"POS vocabulary size: {len(pos_vocab)}")
    
    if len(word_vocab) < 100:
        print("Warning: Very small vocabulary! Consider reducing vocab_min_freq.")
    
    # Create encoders
    word_to_idx = {word: idx for idx, word in enumerate(word_vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    pos_to_idx = {pos: idx for idx, pos in enumerate(pos_vocab)}
    
    # Enhanced encoding with shape validation
    def encode_sequences_robust(tokens_list, pos_list, word_to_idx, pos_to_idx, max_length, add_start_end=False):
        encoded_tokens = []
        encoded_pos = []
        
        for i, (tokens, pos_tags) in enumerate(zip(tokens_list, pos_list)):
            try:
                # Handle empty sequences
                if len(tokens) == 0:
                    tokens = ['<UNK>']
                    pos_tags = ['NOUN']
                
                # Encode tokens
                token_ids = []
                if add_start_end:
                    token_ids.append(word_to_idx['<START>'])
                
                for token in tokens:
                    token_ids.append(word_to_idx.get(token, word_to_idx['<UNK>']))
                
                if add_start_end:
                    token_ids.append(word_to_idx['<END>'])
                
                # Encode POS tags (ensure same length as tokens)
                pos_ids = []
                for j, pos in enumerate(pos_tags):
                    if j < len(tokens):  # Ensure we don't exceed token length
                        pos_ids.append(pos_to_idx.get(pos, pos_to_idx.get('NOUN', 0)))
                
                # Ensure pos_ids matches token length (before start/end tokens)
                while len(pos_ids) < len(tokens):
                    pos_ids.append(pos_to_idx.get('NOUN', 0))
                
                pos_ids = pos_ids[:len(tokens)]  # Trim if too long
                
                # Determine target length
                if add_start_end:
                    target_len = max_length + 2
                    # Pad POS to match token length with start/end
                    pos_ids = [0] + pos_ids + [0]  # Add padding for start/end
                else:
                    target_len = max_length
                
                # Pad sequences
                token_ids = pad_sequences([token_ids], maxlen=target_len, padding='post')[0]
                pos_ids = pad_sequences([pos_ids], maxlen=max_length, padding='post')[0]
                
                encoded_tokens.append(token_ids)
                encoded_pos.append(pos_ids)
                
            except Exception as e:
                print(f"Error encoding sequence {i}: {e}")
                # Fallback: create minimal valid sequence
                if add_start_end:
                    token_ids = [word_to_idx['<START>'], word_to_idx['<UNK>'], word_to_idx['<END>']]
                    token_ids = pad_sequences([token_ids], maxlen=max_length + 2, padding='post')[0]
                else:
                    token_ids = [word_to_idx['<UNK>']]
                    token_ids = pad_sequences([token_ids], maxlen=max_length, padding='post')[0]
                
                pos_ids = [pos_to_idx.get('NOUN', 0)]
                pos_ids = pad_sequences([pos_ids], maxlen=max_length, padding='post')[0]
                
                encoded_tokens.append(token_ids)
                encoded_pos.append(pos_ids)
        
        return np.array(encoded_tokens), np.array(encoded_pos)
    
    print("Encoding sequences...")
    X_tokens, X_pos = encode_sequences_robust(doc_tokens, doc_pos, word_to_idx, pos_to_idx, max_doc_length)
    y_tokens, _ = encode_sequences_robust(summary_tokens, [['NOUN'] * len(tokens) for tokens in summary_tokens], 
                                         word_to_idx, pos_to_idx, max_summary_length, add_start_end=True)
    
    # Validate shapes
    print(f"X_tokens shape: {X_tokens.shape}")
    print(f"X_pos shape: {X_pos.shape}")
    print(f"y_tokens shape: {y_tokens.shape}")
    
    # Create decoder input and output with shape validation
    decoder_input = y_tokens[:, :-1]  # All except last token
    decoder_output = y_tokens[:, 1:]   # All except first token
    
    print(f"Decoder input shape: {decoder_input.shape}")
    print(f"Decoder output shape: {decoder_output.shape}")
    
    # Ensure minimum batch size
    if len(X_tokens) < batch_size:
        print(f"Warning: Dataset size ({len(X_tokens)}) smaller than batch size ({batch_size})")
        batch_size = max(1, len(X_tokens) // 4)
        print(f"Adjusting batch size to: {batch_size}")
    
    # Convert to categorical for output
    try:
        decoder_output = to_categorical(decoder_output, num_classes=len(word_vocab))
        print(f"Decoder output categorical shape: {decoder_output.shape}")
    except Exception as e:
        print(f"Error in to_categorical conversion: {e}")
        return None, None, None
    
    print("Splitting data...")
    # Split data with stratification handling
    try:
        indices = np.arange(len(X_tokens))
        
        if len(indices) < 10:  # Too small for proper split
            print("Dataset too small for train/test split. Using 80/20 split with no validation.")
            split_idx = int(0.8 * len(indices))
            train_idx = indices[:split_idx]
            test_idx = indices[split_idx:]
        else:
            train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
        
        X_tokens_train, X_tokens_test = X_tokens[train_idx], X_tokens[test_idx]
        X_pos_train, X_pos_test = X_pos[train_idx], X_pos[test_idx]
        decoder_input_train, decoder_input_test = decoder_input[train_idx], decoder_input[test_idx]
        decoder_output_train, decoder_output_test = decoder_output[train_idx], decoder_output[test_idx]
        
        print(f"Training samples: {len(train_idx)}, Test samples: {len(test_idx)}")
        
    except Exception as e:
        print(f"Error in data splitting: {e}")
        return None, None, None
    
    print("Building model...")
    
    try:
        # Build encoder-decoder model with POS features and robust architecture
        
        # Encoder
        encoder_token_input = Input(shape=(max_doc_length,), name='encoder_tokens')
        encoder_pos_input = Input(shape=(max_doc_length,), name='encoder_pos')
        
        # Token embedding with dropout for regularization
        token_embedding = Embedding(len(word_vocab), embedding_dim, mask_zero=True)(encoder_token_input)
        
        # POS embedding
        pos_embedding_dim = min(32, len(pos_vocab) // 2)  # Adaptive POS embedding size
        pos_embedding = Embedding(len(pos_vocab), pos_embedding_dim, mask_zero=True)(encoder_pos_input)
        
        # Concatenate embeddings
        from tensorflow.keras.layers import Concatenate, Dropout
        combined_embedding = Concatenate()([token_embedding, pos_embedding])
        combined_embedding = Dropout(0.2)(combined_embedding)
        
        # Encoder LSTM with dropout
        encoder_lstm = LSTM(lstm_units, return_state=True, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        encoder_outputs, state_h, state_c = encoder_lstm(combined_embedding)
        encoder_states = [state_h, state_c]
        
        # Decoder
        decoder_input_layer = Input(shape=(None,), name='decoder_input')
        decoder_embedding = Embedding(len(word_vocab), embedding_dim)(decoder_input_layer)
        decoder_embedding = Dropout(0.2)(decoder_embedding)
        
        decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True, dropout=0.2, recurrent_dropout=0.2)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        
        # Simple attention mechanism
        from tensorflow.keras.layers import Dot, Activation
        
        attention = Dot(axes=[2, 2])([decoder_outputs, encoder_outputs])
        attention = Activation('softmax')(attention)
        
        context = Dot(axes=[2, 1])([attention, encoder_outputs])
        
        # Concatenate context and decoder output
        decoder_combined_context = Concatenate()([context, decoder_outputs])
        decoder_combined_context = Dropout(0.3)(decoder_combined_context)
        
        # Output layer
        output_layer = TimeDistributed(Dense(len(word_vocab), activation='softmax'))
        decoder_outputs = output_layer(decoder_combined_context)
        
        # Define model
        model = Model([encoder_token_input, encoder_pos_input, decoder_input_layer], decoder_outputs)
        
        # Compile model with adaptive learning rate
        initial_lr = 0.001 if len(word_vocab) > 1000 else 0.01
        model.compile(optimizer=Adam(learning_rate=initial_lr), 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
        
        print("Model summary:")
        model.summary()
        
    except Exception as e:
        print(f"Error building model: {e}")
        return None, None, None
    
    print("Training model...")
    
    try:
        # Enhanced training callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=max(5, epochs//10), 
                                     restore_best_weights=True, verbose=1)
        
        # Reduce batch size if memory issues
        effective_batch_size = min(batch_size, len(train_idx))
        
        # Train model with error handling
        history = model.fit(
            [X_tokens_train, X_pos_train, decoder_input_train],
            decoder_output_train,
            batch_size=effective_batch_size,
            epochs=epochs,
            validation_data=([X_tokens_test, X_pos_test, decoder_input_test], decoder_output_test),
            callbacks=[early_stopping],
            verbose=1
        )
        
    except Exception as e:
        print(f"Error during training: {e}")
        return None, None, None
    
    print("Saving model and encoders...")
    
    try:
        # Save model
        model.save(model_save_path)
        
        # Save encoders and metadata with comprehensive info
        encoders = {
            'word_to_idx': word_to_idx,
            'idx_to_word': idx_to_word,
            'pos_to_idx': pos_to_idx,
            'word_vocab': word_vocab,
            'pos_vocab': pos_vocab,
            'max_doc_length': max_doc_length,
            'max_summary_length': max_summary_length,
            'embedding_dim': embedding_dim,
            'lstm_units': lstm_units,
            'vocab_size': len(word_vocab),
            'pos_vocab_size': len(pos_vocab),
            'spacy_model': 'en_core_web_sm',
            'training_samples': len(train_idx),
            'vocab_min_freq': vocab_min_freq,
            'data_stats': {
                'avg_doc_length': np.mean([len(tokens) for tokens in doc_tokens]),
                'avg_summary_length': np.mean([len(tokens) for tokens in summary_tokens]),
                'total_samples': len(doc_tokens)
            }
        }
        
        with open(encoders_save_path, 'wb') as f:
            pickle.dump(encoders, f)
        
        print(f"Model saved to: {model_save_path}")
        print(f"Encoders saved to: {encoders_save_path}")
        print("Training completed successfully!")
        
        # Print training summary
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        print(f"Final training loss: {final_loss:.4f}")
        print(f"Final validation loss: {final_val_loss:.4f}")
        
    except Exception as e:
        print(f"Error saving model: {e}")
        return model, history, None  # Return what we can
    
    return model, history, encoders

# Example usage:
"""
# Train and save the model with robust edge case handling
model, history, encoders = create_and_train_pos_lstm_summarizer(
    csv_file_path='your_data.csv',
    document_column='document',
    summary_column='summary',
    epochs=30,
    batch_size=16,
    model_save_path='my_summarizer_model.h5',
    encoders_save_path='my_encoders.pkl',
    min_doc_length=5,           # Minimum document length
    min_summary_length=2,       # Minimum summary length
    vocab_min_freq=2,          # Minimum word frequency
    handle_empty_strings=True,  # Handle empty/null strings
    auto_adjust_lengths=True    # Auto-adjust max lengths based on data
)

# The model and encoders are now saved to disk with comprehensive edge case handling
"""
import os
import sys

def main():
    """
    Main function to train the POS-LSTM summarization model
    """
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate to project root (AutoSummerizer CLI)
    project_root = os.path.dirname(current_dir)  # Go up one level from current script location
    
    # Construct paths
    csv_file_path = os.path.join(project_root, 'data', 'summaries.csv')
    model_save_path = os.path.join(project_root, 'models', 'trained_summarizer_model.h5')
    encoders_save_path = os.path.join(project_root, 'models', 'model_encoders.pkl')
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Project root: {project_root}")
    print(f"CSV file path: {csv_file_path}")
    print(f"Model save path: {model_save_path}")
    print(f"Encoders save path: {encoders_save_path}")
    
    # Check if CSV file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at {csv_file_path}")
        print("Please ensure the summaries.csv file exists in the data directory.")
        return
    
    # Import the training function (assuming it's in the same file or imported)
    # If the function is in a separate file, uncomment and modify the import below:
    # from your_module import create_and_train_pos_lstm_summarizer
    
    try:
        print("Starting model training...")
        print("="*50)
        
        # Call the training function with robust parameters
        model, history, encoders = create_and_train_pos_lstm_summarizer(
            csv_file_path=csv_file_path,
            document_column='document',           # Adjust column name if different
            summary_column='summary',             # Adjust column name if different
            max_doc_length=500,                   # Will auto-adjust if needed
            max_summary_length=100,               # Will auto-adjust if needed
            embedding_dim=128,
            lstm_units=256,
            epochs=30,                            # Adjust based on your needs
            batch_size=16,                        # Will auto-adjust for small datasets
            test_size=0.2,
            model_save_path=model_save_path,
            encoders_save_path=encoders_save_path,
            # Edge case handling parameters
            min_doc_length=5,
            min_summary_length=2,
            vocab_min_freq=2,
            handle_empty_strings=True,
            auto_adjust_lengths=True
        )
        
        if model is not None:
            print("="*50)
            print("Training completed successfully!")
            print(f"Model saved to: {model_save_path}")
            print(f"Encoders saved to: {encoders_save_path}")
            
            # Print some training statistics if available
            if history is not None:
                final_loss = history.history['loss'][-1]
                final_val_loss = history.history['val_loss'][-1]
                print(f"Final training loss: {final_loss:.4f}")
                print(f"Final validation loss: {final_val_loss:.4f}")
                
            if encoders is not None:
                print(f"Vocabulary size: {encoders['vocab_size']}")
                print(f"Training samples: {encoders['training_samples']}")
                print(f"Average document length: {encoders['data_stats']['avg_doc_length']:.1f}")
                print(f"Average summary length: {encoders['data_stats']['avg_summary_length']:.1f}")
        else:
            print("Training failed. Please check the error messages above.")
            
    except Exception as e:
        print(f"Error during training: {str(e)}")
        print("Please check your data file and try again.")
        import traceback
        traceback.print_exc()

def check_dependencies():
    """
    Check if required dependencies are installed
    """
    # Map of package names to their import names
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'spacy': 'spacy',
        'tensorflow': 'tensorflow',
        'scikit-learn': 'sklearn',  # scikit-learn imports as sklearn
        'pickle': 'pickle'
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    # Check spaCy model
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except OSError:
        print("spaCy English model not found.")
        print("Please install it using:")
        print("python -m spacy download en_core_web_sm")
        return False
    
    return True

if __name__ == "__main__":
    print("POS-LSTM Summarization Model Training")
    print("="*40)
    
    # Check dependencies first
    if not check_dependencies():
        print("Please install missing dependencies before running.")
        sys.exit(1)
    
    # Run the main training function
    main()
    
    print("\nScript execution completed.")