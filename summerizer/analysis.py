# Content retention analysis using spaCy

import pandas as pd
import spacy

# Load spacy model
nlp = spacy.load("en_core_web_sm")

# Load dataset
df = pd.read_csv("data/summaries.csv")

# Sample a single row
sample = df.sample(1).iloc[0]
doc_text = sample['document']
summary_text = sample['summary']

# NLP processing
doc_doc = nlp(doc_text)
sum_doc = nlp(summary_text)

# Important entity labels to track
important_labels = {"PERSON", "ORG", "DATE", "TIME", "MONEY", "PERCENT", "CARDINAL", "QUANTITY"}

# Extract entities from both
doc_ents = set([ent.text for ent in doc_doc.ents if ent.label_ in important_labels])
sum_ents = set([ent.text for ent in sum_doc.ents if ent.label_ in important_labels])

# Compute retention
retained = doc_ents & sum_ents
retention_ratio = len(retained) / len(doc_ents) if doc_ents else 0

# Display results
print(f"📝 Document entities: {doc_ents}")
print(f"✂️ Summary entities: {sum_ents}")
print(f"✅ Retained entities: {retained}")
print(f"📊 Retention Score: {retention_ratio:.2f}")
