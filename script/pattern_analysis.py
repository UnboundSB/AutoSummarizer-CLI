import pandas as pd
import spacy
MODEL = 'en_core_web_sm'

df =pd.read_csv('data/summaries.csv')
sampled = df.sample(1)[['document', 'summary']]
word_count=df['document'].str.split().str.len()
nlp = spacy.load(MODEL)

text=sampled['document'].iloc[0]
summary=sampled['summary'].iloc[0]

doc = nlp(text)
sumdoc=nlp(summary)

entities=doc.ents
summary_ents=sumdoc.ents

named_entities=[(ent.text,ent.label_) for ent in entities]
summerized_named_entities=[(ent.text, ent.label_) for ent in summary_ents]

noun_phrases=[token.text for token in doc.noun_chunks]
summary_noun_phrase=[token.text for token in sumdoc.noun_chunks]


print(set(named_entities) & set(summerized_named_entities))
print('\n\n\n\n\n')
print(set(noun_phrases) & set(summary_noun_phrase))