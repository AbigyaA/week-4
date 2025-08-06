import pandas as pd
import re

# Load the cleaned Amharic messages from Task 1
df = pd.read_json("preprocessed_telegram_data.json", lines=True)

# Select first 50 messages for labeling
subset = df[['clean_text']].dropna().head(50).reset_index(drop=True)
subset.columns = ['message']
def tokenize(text):
    # Remove punctuation and split
    text = re.sub(r'[፣።፡፥፦፧፨]', ' ', text)
    tokens = text.strip().split()
    return tokens

subset['tokens'] = subset['message'].apply(tokenize)
def manual_label_message(tokens):
    print("\nMessage:")
    print(" ".join(tokens))
    print("Enter labels (B-Product, I-Product, B-LOC, I-LOC, B-PRICE, I-PRICE, O):")

    labels = []
    for token in tokens:
        label = input(f"{token}: ").strip()
        if label not in ['B-Product', 'I-Product', 'B-LOC', 'I-LOC', 'B-PRICE', 'I-PRICE', 'O']:
            label = 'O'  # fallback
        labels.append((token, label))
    return labels
all_labeled = []

for i, row in subset.iterrows():
    print(f"\n--- Message {i+1}/{len(subset)} ---")
    tokens = row['tokens']
    labeled = manual_label_message(tokens)
    all_labeled.append(labeled)

# Save to CoNLL format
with open("amharic_ner_labeled.conll", "w", encoding='utf-8') as f:
    for sentence in all_labeled:
        for token, label in sentence:
            f.write(f"{token}\t{label}\n")
        f.write("\n")  # blank line between messages
