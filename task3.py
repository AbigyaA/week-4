import pandas as pd

# Load CoNLL format
def read_conll(file_path):
    sentences, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        tokens, tags = [], []
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens, tags = [], []
            else:
                token, tag = line.split('\t')
                tokens.append(token)
                tags.append(tag)
    return sentences, labels

sentences, tags = read_conll("amharic_ner_labeled.conll")
from datasets import Dataset

data = [{"tokens": s, "ner_tags": t} for s, t in zip(sentences, tags)]
dataset = Dataset.from_list(data)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
# Unique label list
unique_labels = sorted(set(tag for sent in tags for tag in sent))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

# Convert tags to IDs
def encode_labels(example):
    example["label_ids"] = [label2id[tag] for tag in example["ner_tags"]]
    return example

dataset = dataset.map(encode_labels)
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_checkpoint = "Davlan/bert-tiny-amharic"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
)
from transformers import DataCollatorForTokenClassification
import numpy as np

def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
    word_ids = tokenized_inputs.word_ids()
    label_ids = []
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            label_ids.append(label2id[example["ner_tags"][word_idx]])
        else:
            label_ids.append(label2id[example["ner_tags"][word_idx]])
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = label_ids
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=False)
data_collator = DataCollatorForTokenClassification(tokenizer)
from transformers import Trainer, TrainingArguments
import evaluate

metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return metric.compute(predictions=true_predictions, references=true_labels)

training_args = TrainingArguments(
    output_dir="./amharic-ner",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
model.save_pretrained("finetuned_amharic_ner")
tokenizer.save_pretrained("finetuned_amharic_ner")
