def load_model_for_stage(stage: int, model_dir: str = "models"):
    """Load model and vocabularies for a specific stage"""
    model_path = f"{model_dir}/lstm_summarizer_stage_{stage}.keras"
    vocab_path = f"{model_dir}/vocabularies_stage_{stage}.json"
    
    # Also check for old .h5 format
    if not os.path.exists(model_path):
        model_path = f"{model_dir}/lstm_summarizer_stage_{stage}.h5"
    
    if os.path.exists(model_path) and os.path.exists(vocab_path):
        try:
            # Load model
            model = tf.keras.models.load_model(model_path, compile=False)
            
            # RECOMPILE with fresh optimizer
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Load vocabularies
            with open(vocab_path, 'r') as f:
                vocab_data = json.load(f)
            
            vocabularies = {
                'token_vocab': vocab_data['token_vocab'],
                'pos_vocab': vocab_data['pos_vocab'],
                'dep_vocab': vocab_data['dep_vocab'],
                'ent_vocab': vocab_data['ent_vocab']
            }
            
            # Create reverse vocabulary for decoding
            vocabularies['reverse_token_vocab'] = {v: k for k, v in vocabularies['token_vocab'].items()}
            
            print(f"✅ Stage {stage}% model loaded successfully")
            return model, vocabularies
            
        except Exception as e:
            print(f"❌ Error loading stage {stage}%: {e}")
            return None, None
    else:
        print(f"❌ Stage {stage}% model files not found")
        return None, None

def summarize_with_stage_model(text: str, model, vocabularies, max_summary_length: int = 256) -> str:
    """Summarize text using a specific stage model"""
    
    # Validate input
    if not text or not isinstance(text, str):
        return "❌ Error: Invalid text input"
    
    if len(text.strip()) < 10:
        return "⚠️ Text too short for summarization"
    
    try:
        # Set global vocabularies temporarily
        global VOCABULARIES
        original_vocab = VOCABULARIES
        VOCABULARIES = vocabularies
        
        # Convert text to model input format
        input_sequences = text_to_inference_sequences(text, max_len=512)
        
        # Prepare input for model (add batch dimension)
        X_tokens = np.expand_dims(input_sequences[:, 0], 0)
        X_pos = np.expand_dims(input_sequences[:, 1], 0)
        X_deps = np.expand_dims(input_sequences[:, 2], 0)
        X_ents = np.expand_dims(input_sequences[:, 3], 0)
        
        # Generate summary using the model
        model_output = model.predict([X_tokens, X_pos, X_deps, X_ents], verbose=0)
        
        # Convert model output back to text
        summary = sequences_to_text(model_output, top_k=5)
        
        # Post-process summary
        summary = post_process_summary(summary, max_summary_length)
        
        # Restore original vocabularies
        VOCABULARIES = original_vocab
        
        return summary
        
    except Exception as e:
        # Restore original vocabularies on error
        VOCABULARIES = original_vocab
        return f"❌ Error: {str(e)}"

def test_multiple_stage_models():
    """Test models from stages 10, 20, 30, 40, 50 on real data"""
    
    print("\n🔬 MULTI-STAGE MODEL COMPARISON")
    print("=" * 80)
    
    # Load test documents from CSV
    test_documents = load_test_documents_from_csv(num_docs=20)
    
    if not test_documents:
        print("❌ Could not load test documents from CSV")
        return
    
    # Test stages
    test_stages = [10, 20, 30, 40, 50]
    stage_results = {}
    
    for stage in test_stages:
        print(f"\n🎯 TESTING STAGE {stage}% MODEL")
        print("-" * 60)
        
        # Load model for this stage
        model, vocabularies = load_model_for_stage(stage)
        
        if model is None:
            print(f"⚠️ Stage {stage}% model not available, skipping...")
            continue
        
        # Test this stage on all documents
        stage_times = []
        stage_summaries = []
        
        for i, (document, ground_truth) in enumerate(test_documents, 1):
            print(f"📄 Document {i}/20 - Stage {stage}%", end=" ")
            
            # Time the summarization
            start_time = time.time()
            summary = summarize_with_stage_model(document.strip(), model, vocabularies)
            end_time = time.time()
            
            processing_time = end_time - start_time
            stage_times.append(processing_time)
            stage_summaries.append(summary)
            
            print(f"⏱️ {processing_time:.3f}s")
        
        # Calculate stage metrics
        avg_time = np.mean(stage_times)
        total_time = sum(stage_times)
        
        stage_results[stage] = {
            'times': stage_times,
            'summaries': stage_summaries,
            'avg_time': avg_time,
            'total_time': total_time
        }
        
        print(f"\n📊 Stage {stage}% Results:")
        print(f"   ⏱️ Average Time: {avg_time:.3f} seconds")
        print(f"   ⏱️ Total Time: {total_time:.3f} seconds")
        print(f"   🚀 Speed: {len(test_documents)/total_time:.2f} docs/sec")
        
        # Show sample summary
        if stage_summaries:
            print(f"   📋 Sample Summary: {stage_summaries[0][:100]}{'...' if len(stage_summaries[0]) > 100 else ''}")
    
    # Compare all stages
    print(f"\n📊 FINAL COMPARISON - ALL STAGES")
    print("=" * 80)
    
    print(f"{'Stage':<8} {'Avg Time':<12} {'Total Time':<12} {'Speed (docs/s)':<15} {'Available':<10}")
    print("-" * 70)
    
    for stage in test_stages:
        if stage in stage_results:
            result = stage_results[stage]
            print(f"{stage}%{'':<5} {result['avg_time']:.3f}s{'':<5} {result['total_time']:.3f}s{'':<5} "
                  f"{len(test_documents)/result['total_time']:.2f}{'':<11} ✅")
        else:
            print(f"{stage}%{'':<5} {'N/A':<10} {'N/A':<10} {'N/A':<13} ❌")
    
    # Find best performing stage
    if stage_results:
        best_stage = min(stage_results.keys(), key=lambda s: stage_results[s]['avg_time'])
        worst_stage = max(stage_results.keys(), key=lambda s: stage_results[s]['avg_time'])
        
        print(f"\n🏆 PERFORMANCE WINNER:")
        print(f"   🥇 Fastest: Stage {best_stage}% ({stage_results[best_stage]['avg_time']:.3f}s avg)")
        print(f"   🐌 Slowest: Stage {worst_stage}% ({stage_results[worst_stage]['avg_time']:.3f}s avg)")
        
        # Show quality comparison with sample summaries
        print(f"\n📋 QUALITY COMPARISON (Document 1):")
        print("-" * 60)
        for stage in sorted(stage_results.keys()):
            summary = stage_results[stage]['summaries'][0] if stage_results[stage]['summaries'] else "No summary"
            print(f"Stage {stage}%: {summary[:120]}{'...' if len(summary) > 120 else ''}")
    
    return stage_results

def load_test_documents_from_csv(csv_path: str = "summaries.csv", num_docs: int = 20):
    """Load test documents from the later 50% of the dataset"""
    try:
        import pandas as pd
        
        if not os.path.exists(csv_path):
            print(f"⚠️ CSV file not found: {csv_path}")
            return None
        
        # Load the dataset
        df = pd.read_csv(csv_path)
        print(f"📊 Loaded dataset: {len(df)} total samples")
        
        # Get the later 50% of data (unseen during training)
        start_idx = len(df) // 2  # 50% mark
        test_df = df.iloc[start_idx:].copy()
        
        print(f"📊 Using later 50% for testing: {len(test_df)} samples available")
        
        # Randomly sample documents from the later 50%
        if len(test_df) < num_docs:
            print(f"⚠️ Only {len(test_df)} samples available, using all of them")
            sample_df = test_df
        else:
            sample_df = test_df.sample(n=num_docs, random_state=42)
        
        print(f"🎯 Selected {len(sample_df)} documents for performance testing")
        
        # Check column names
        doc_col = None
        summary_col = None
        
        # Try common column name variations
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in ['document', 'documents', 'text', 'article', 'content']:
                doc_col = col
            elif col_lower in ['summary', 'summaries', 'abstract', 'synopsis']:
                summary_col = col
        
        if doc_col is None or summary_col is None:
            print(f"❌ Could not identify document/summary columns")
            print(f"Available columns: {df.columns.tolist()}")
            return None
        
        print(f"📊 Using columns: '{doc_col}' (documents) and '{summary_col}' (summaries)")
        
        # Extract document-summary pairs
        test_pairs = []
        for _, row in sample_df.iterrows():
            document = str(row[doc_col])
            ground_truth = str(row[summary_col])
            
            # Skip if either is too short or invalid
            if len(document.strip()) < 50 or len(ground_truth.strip()) < 10:
                continue
                
            test_pairs.append((document, ground_truth))
        
        print(f"✅ Prepared {len(test_pairs)} valid document-summary pairs for testing")
        return test_pairs
        
    except Exception as e:
        print(f"❌ Error loading test documents: {e}")
        return None# summarize.py - Text Summarization Inference using Trained LSTM Model
# Functional Programming Approach

import tensorflow as tf
import spacy
import numpy as np
import json
import os
from typing import Optional

# Global variables for model and vocabularies (loaded once)
MODEL = None
VOCABULARIES = None
NLP = None

def load_model_components(model_path: str = "D:\Projects\AutoSummerizer CLI\models\stage4\lstm_summarizer_stage_4.keras", 
                         vocab_path: str = "D:\Projects\AutoSummerizer CLI\models\stage4\vocabularies_stage_4.json"):
    """Load the trained model and vocabularies"""
    global MODEL, VOCABULARIES, NLP
    
    try:
        # Load spaCy model
        if NLP is None:
            NLP = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded")
        
        # Load TensorFlow model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        MODEL = tf.keras.models.load_model(model_path, compile=False)
        print(f"✅ Neural network model loaded from {model_path}")
        
        # Load vocabularies
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
        
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        
        VOCABULARIES = {
            'token_vocab': vocab_data['token_vocab'],
            'pos_vocab': vocab_data['pos_vocab'],
            'dep_vocab': vocab_data['dep_vocab'],
            'ent_vocab': vocab_data['ent_vocab']
        }
        
        # Create reverse vocabularies for decoding
        VOCABULARIES['reverse_token_vocab'] = {v: k for k, v in VOCABULARIES['token_vocab'].items()}
        
        print(f"✅ Vocabularies loaded - Token vocab size: {len(VOCABULARIES['token_vocab'])}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model components: {e}")
        return False

def extract_spacy_features_inference(text: str, max_tokens: int = 512) -> dict:
    """Extract spaCy features for inference"""
    if NLP is None:
        raise RuntimeError("spaCy model not loaded. Call load_model_components() first.")
    
    # Limit text length to avoid memory issues
    doc = NLP(text[:10000])
    
    # Extract features
    tokens = []
    pos_tags = []
    dep_tags = []
    ent_labels = []
    
    for token in doc[:max_tokens]:
        tokens.append(token.text.lower())
        pos_tags.append(token.pos_)
        dep_tags.append(token.dep_)
        ent_labels.append(token.ent_type_ if token.ent_type_ else "O")
    
    return {
        'tokens': tokens,
        'pos_tags': pos_tags,
        'dep_tags': dep_tags,
        'ent_labels': ent_labels
    }

def text_to_inference_sequences(text: str, max_len: int = 512) -> np.ndarray:
    """Convert text to numerical sequences for model inference"""
    if VOCABULARIES is None:
        raise RuntimeError("Vocabularies not loaded. Call load_model_components() first.")
    
    # Extract features
    features = extract_spacy_features_inference(text, max_len)
    
    # Convert to sequences using vocabularies
    token_seq = [VOCABULARIES['token_vocab'].get(token, 1) for token in features['tokens']]
    pos_seq = [VOCABULARIES['pos_vocab'].get(pos, 0) for pos in features['pos_tags']]
    dep_seq = [VOCABULARIES['dep_vocab'].get(dep, 0) for dep in features['dep_tags']]
    ent_seq = [VOCABULARIES['ent_vocab'].get(ent, 0) for ent in features['ent_labels']]
    
    # Pad sequences to max_len
    token_seq = token_seq[:max_len] + [0] * (max_len - len(token_seq))
    pos_seq = pos_seq[:max_len] + [0] * (max_len - len(pos_seq))
    dep_seq = dep_seq[:max_len] + [0] * (max_len - len(dep_seq))
    ent_seq = ent_seq[:max_len] + [0] * (max_len - len(ent_seq))
    
    # Combine into single array
    return np.array([token_seq, pos_seq, dep_seq, ent_seq]).T

def sequences_to_text(sequences: np.ndarray, top_k: int = 5) -> str:
    """Convert model output sequences back to text"""
    if VOCABULARIES is None:
        raise RuntimeError("Vocabularies not loaded. Call load_model_components() first.")
    
    reverse_vocab = VOCABULARIES['reverse_token_vocab']
    
    # Handle model output (probability distribution)
    if len(sequences.shape) == 3:  # (batch, seq_len, vocab_size)
        # Take the most likely token at each position
        token_indices = np.argmax(sequences[0], axis=-1)
    else:
        token_indices = sequences
    
    # Convert indices to tokens
    tokens = []
    for idx in token_indices:
        token = reverse_vocab.get(int(idx), '<UNK>')
        if token in ['<PAD>', '<UNK>']:
            continue
        if token == '</s>' or token == '<EOS>':  # End of sequence
            break
        tokens.append(token)
    
    # Join tokens and clean up
    summary = ' '.join(tokens)
    
    # Basic cleanup
    summary = summary.replace(' .', '.').replace(' ,', ',').replace(' !', '!')
    summary = summary.replace(' ?', '?').replace(' ;', ';').replace(' :', ':')
    
    return summary.strip()

def summarize(text: str, max_summary_length: int = 256) -> str:
    """
    Main summarization function - takes text input and returns summary
    
    Args:
        text (str): Input text to summarize
        max_summary_length (int): Maximum length of generated summary
    
    Returns:
        str: Generated summary
    """
    
    # Check if model components are loaded
    if MODEL is None or VOCABULARIES is None:
        print("🔄 Loading model components...")
        success = load_model_components()
        if not success:
            return "❌ Error: Could not load model. Please check model files."
    
    # Validate input
    if not text or not isinstance(text, str):
        return "❌ Error: Please provide valid text input."
    
    if len(text.strip()) < 10:
        return "⚠️ Warning: Input text too short for meaningful summarization."
    
    try:
        print(f"📝 Processing text ({len(text)} characters)...")
        
        # Convert text to model input format
        input_sequences = text_to_inference_sequences(text, max_len=512)
        
        # Prepare input for model (add batch dimension)
        X_tokens = np.expand_dims(input_sequences[:, 0], 0)
        X_pos = np.expand_dims(input_sequences[:, 1], 0)
        X_deps = np.expand_dims(input_sequences[:, 2], 0)
        X_ents = np.expand_dims(input_sequences[:, 3], 0)
        
        print("🧠 Generating summary...")
        
        # Generate summary using the model
        model_output = MODEL.predict([X_tokens, X_pos, X_deps, X_ents], verbose=0)
        
        # Convert model output back to text
        summary = sequences_to_text(model_output, top_k=5)
        
        # Post-process summary
        summary = post_process_summary(summary, max_summary_length)
        
        print("✅ Summary generated successfully!")
        return summary
        
    except Exception as e:
        error_msg = f"❌ Error during summarization: {str(e)}"
        print(error_msg)
        return error_msg

def post_process_summary(summary: str, max_length: int = 256) -> str:
    """Post-process the generated summary"""
    
    # Remove extra whitespace
    summary = ' '.join(summary.split())
    
    # Truncate to max length while preserving sentence boundaries
    if len(summary) > max_length:
        sentences = summary.split('.')
        result = ""
        for sentence in sentences:
            if len(result + sentence + '.') <= max_length:
                result += sentence + '.'
            else:
                break
        summary = result.strip()
    
    # Ensure it ends with proper punctuation
    if summary and not summary.endswith(('.', '!', '?')):
        summary += '.'
    
    # Capitalize first letter
    if summary:
        summary = summary[0].upper() + summary[1:] if len(summary) > 1 else summary.upper()
    
    return summary

def batch_summarize(texts: list, max_summary_length: int = 256) -> list:
    """Summarize multiple texts efficiently"""
    summaries = []
    
    print(f"📚 Summarizing {len(texts)} texts...")
    
    for i, text in enumerate(texts, 1):
        print(f"📄 Processing text {i}/{len(texts)}")
        summary = summarize(text, max_summary_length)
        summaries.append(summary)
    
    return summaries

def test_summarization():
    """Test function to verify the model works"""
    
    sample_text = """
    Artificial intelligence (AI) has rapidly transformed various industries over the past decade. 
    Machine learning algorithms, particularly deep learning neural networks, have enabled computers 
    to perform tasks that were previously thought to require human intelligence. These include 
    image recognition, natural language processing, speech recognition, and decision-making. 
    Companies across sectors like healthcare, finance, transportation, and technology have 
    integrated AI systems to improve efficiency, reduce costs, and enhance user experiences. 
    However, the rapid advancement of AI has also raised concerns about job displacement, 
    privacy, and ethical considerations. As AI continues to evolve, society must balance 
    the benefits of technological progress with responsible development and deployment practices.
    """
    
    print("🧪 Testing summarization with sample text...")
    print(f"📄 Input length: {len(sample_text)} characters")
    
    summary = summarize(sample_text)
    
    print(f"\n📋 Generated Summary:")
    print(f"📄 Output length: {len(summary)} characters")
    print("-" * 50)
    print(summary)
    print("-" * 50)
    
    return summary

# Main execution
if __name__ == "__main__":
    import time
    
    print("🤖 LSTM Text Summarizer - Inference Module")
    print("=" * 60)
    
    # Load model components
    success = load_model_components()
    
    if success:
        print("\n🧪 Running test summarization...")
        test_summarization()
        
        print("\n⏱️ Performance Testing - Multi-Document Summarization")
        print("=" * 60)
        
        # Test documents (2-3 sample paragraphs)
        test_documents = [
            """
            Climate change represents one of the most pressing challenges of our time. Global temperatures 
            have risen significantly over the past century, primarily due to human activities such as 
            burning fossil fuels, deforestation, and industrial processes. The consequences include melting 
            ice caps, rising sea levels, extreme weather events, and disruptions to ecosystems worldwide. 
            Scientists emphasize the urgent need for immediate action to reduce greenhouse gas emissions 
            and transition to renewable energy sources. International cooperation through agreements like 
            the Paris Climate Accord aims to limit global warming and protect the planet for future generations.
            """,
            
            """
            The rapid advancement of artificial intelligence has revolutionized numerous industries and 
            aspects of daily life. Machine learning algorithms can now diagnose diseases, predict market 
            trends, autonomous vehicles navigate complex environments, and virtual assistants understand 
            natural language. Deep learning neural networks have achieved remarkable breakthroughs in 
            computer vision, natural language processing, and game playing. However, these developments 
            also raise important questions about job displacement, privacy, algorithmic bias, and the 
            ethical implications of AI decision-making. Ensuring responsible AI development requires 
            careful consideration of these challenges alongside continued innovation.
            """,
            
            """
            Space exploration has entered a new era with private companies joining government agencies 
            in pushing the boundaries of human knowledge and capability. Recent missions to Mars have 
            provided unprecedented insights into the planet's geology and potential for past life. 
            The International Space Station continues to serve as a platform for scientific research 
            and international collaboration. Meanwhile, ambitious projects like lunar bases and Mars 
            colonization are transitioning from science fiction to achievable goals. These endeavors 
            not only advance our understanding of the universe but also drive technological innovations 
            that benefit life on Earth, from satellite communications to medical devices.
            """
        ]
        
        # Time each summarization
        processing_times = []
        summaries = []
        
        print(f"\n📚 Processing {len(test_documents)} documents...\n")
        
        for i, document in enumerate(test_documents, 1):
            print(f"📄 Document {i}:")
            print("-" * 40)
            
            # Time the summarization
            start_time = time.time()
            summary = summarize(document.strip())
            end_time = time.time()
            
            processing_time = end_time - start_time
            processing_times.append(processing_time)
            summaries.append(summary)
            
            # Display results
            print(f"⏱️  Processing Time: {processing_time:.3f} seconds")
            print(f"📋 Summary: {summary}")
            print(f"📏 Original Length: {len(document.strip())} characters")
            print(f"📏 Summary Length: {len(summary)} characters")
            print(f"📊 Compression Ratio: {len(summary)/len(document.strip()):.2%}")
            print()
        
        # Calculate and display average
        avg_time = np.mean(processing_times)
        total_time = sum(processing_times)
        
        print("📊 PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"📄 Documents Processed: {len(test_documents)}")
        print(f"⏱️  Individual Times: {[f'{t:.3f}s' for t in processing_times]}")
        print(f"⏱️  Total Time: {total_time:.3f} seconds")
        print(f"⏱️  Average Time per Document: {avg_time:.3f} seconds")
        print(f"🚀 Processing Speed: {len(test_documents)/total_time:.2f} documents/second")
        
        print(f"\n📋 ALL SUMMARIES:")
        print("-" * 40)
        for i, summary in enumerate(summaries, 1):
            print(f"{i}. {summary}")
        print("-" * 40)
        
        print("\n🎯 Model ready! Use summarize(text) function for your own text.")
        print("📖 Example usage:")
        print("   summary = summarize('Your long text here...')")
        print("   summaries = batch_summarize(['text1', 'text2', 'text3'])")
        
    else:
        print("❌ Failed to load model. Please check that these files exist:")
        print("   - lstm_summarizer_stage_50.keras")
        print("   - vocabularies_stage_50.json")