# summarize_benchmark.py - Benchmark multiple trained LSTM summarizers
import os
import time
import json
import numpy as np
import tensorflow as tf
import spacy

# Globals
MODEL = None
VOCABULARIES = None
NLP = None

def get_stage_paths(stage_num):
    """Return model and vocab paths for a given stage."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    stage_dir = os.path.join(base_dir, f"stage{stage_num}")
    model_path = os.path.join(stage_dir, f"lstm_summarizer_stage_{stage_num}.keras")
    vocab_path = os.path.join(stage_dir, f"vocabularies_stage_{stage_num}.json")
    return model_path, vocab_path

def load_model_components(stage_num):
    """Load the trained model and vocabularies for a stage."""
    global MODEL, VOCABULARIES, NLP
    try:
        if NLP is None:
            NLP = spacy.load("en_core_web_sm")
        model_path, vocab_path = get_stage_paths(stage_num)
        MODEL = tf.keras.models.load_model(model_path, compile=False)
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        VOCABULARIES = {
            'token_vocab': vocab_data['token_vocab'],
            'pos_vocab': vocab_data['pos_vocab'],
            'dep_vocab': vocab_data['dep_vocab'],
            'ent_vocab': vocab_data['ent_vocab'],
            'reverse_token_vocab': {v: k for k, v in vocab_data['token_vocab'].items()}
        }
        return True
    except Exception as e:
        print(f"[Stage {stage_num}] Error loading: {e}")
        return False

def extract_features(text, max_tokens=512):
    """Extract spaCy features."""
    doc = NLP(text[:10000])
    tokens, pos_tags, dep_tags, ent_labels = [], [], [], []
    for token in doc[:max_tokens]:
        tokens.append(token.text.lower())
        pos_tags.append(token.pos_)
        dep_tags.append(token.dep_)
        ent_labels.append(token.ent_type_ if token.ent_type_ else "O")
    return tokens, pos_tags, dep_tags, ent_labels

def text_to_sequences(text, max_len=512):
    """Convert text to model input arrays."""
    tokens, pos_tags, dep_tags, ent_labels = extract_features(text, max_len)
    token_seq = [VOCABULARIES['token_vocab'].get(tok, 1) for tok in tokens]
    pos_seq = [VOCABULARIES['pos_vocab'].get(pos, 0) for pos in pos_tags]
    dep_seq = [VOCABULARIES['dep_vocab'].get(dep, 0) for dep in dep_tags]
    ent_seq = [VOCABULARIES['ent_vocab'].get(ent, 0) for ent in ent_labels]
    # Pad
    pad = lambda seq: seq[:max_len] + [0] * (max_len - len(seq))
    return np.array([pad(token_seq), pad(pos_seq), pad(dep_seq), pad(ent_seq)]).T

def sequences_to_text(seq):
    """Convert predicted sequence back to text."""
    if len(seq.shape) == 3:  # (1, seq_len, vocab_size)
        token_indices = np.argmax(seq[0], axis=-1)
    else:
        token_indices = seq
    reverse_vocab = VOCABULARIES['reverse_token_vocab']
    tokens = []
    for idx in token_indices:
        tok = reverse_vocab.get(int(idx), '<UNK>')
        if tok in ['<PAD>', '<UNK>']: continue
        if tok in ['</s>', '<EOS>']: break
        tokens.append(tok)
    return ' '.join(tokens).replace(' .', '.').replace(' ,', ',')

def summarize(text):
    """Run summarization on text."""
    seqs = text_to_sequences(text, 512)
    X_tokens = np.expand_dims(seqs[:, 0], 0)
    X_pos = np.expand_dims(seqs[:, 1], 0)
    X_deps = np.expand_dims(seqs[:, 2], 0)
    X_ents = np.expand_dims(seqs[:, 3], 0)
    output = MODEL.predict([X_tokens, X_pos, X_deps, X_ents], verbose=0)
    return sequences_to_text(output)

def benchmark(stages, texts):
    """Test multiple stages on given texts."""
    results = []
    for stage in stages:
        print(f"\n=== Stage {stage} ===")
        if not load_model_components(stage):
            continue
        times = []
        for t in texts:
            start = time.time()
            summary = summarize(t)
            times.append(time.time() - start)
            print(f"- Text len: {len(t)}, Summary len: {len(summary)}")
            print(f"  text{t}\n\n\n\n\n\n\n\n\n\n\n\n\nn\n\n\n\nSummary: {summary}...\n")
        avg_time = np.mean(times)
        results.append((stage, avg_time))
    return results

if __name__ == "__main__":
    sample_texts = [
        """Model memorized patterns like “<PAD>” or ultra-short generic phrases because it saw too many similar examples without enough variation.

In seq2seq with sparse vocab, this happens if training targets are short on average, or if teacher forcing ratio stays high and the model never learns long output generation.

Inference decoding issue

If your decoding is just argmax instead of beam search or top-k sampling, the model may always pick the “safe” early-ending token (e.g., <EOS>).

This is especially visible when training data grows but you don’t adjust inference max length or decoding strategy.

Vocab/sequence length mismatch

If max_summary_length during training is short, the model literally can’t output long sequences.

Or if during inference you’re cutting off too early (max_len too small in input or output).

Given your case, my bet is #2 or #3 — your summarize() uses greedy decoding with no beam search, so the model probably learned to bail out early when uncertain.
If the training set size increased but variation in summaries didn’t grow proportionally, this can make the “safe” short output more appealing for the loss function.

If you want, I can tweak your inference to beam search or top-k sampling so we can check if the length shortfall is just decoding, not actual model capacity.""",
        """Climate change represents one of the most pressing challenges of our time..."""
    ]
    stages_to_test = [10, 20, 30, 35, 40, 50]
    results = benchmark(stages_to_test, sample_texts)
    print("\n=== BENCHMARK RESULTS ===")
    for stage, avg_t in results:
        print(f"Stage {stage}: {avg_t:.3f} sec avg")
