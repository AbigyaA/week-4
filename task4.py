model_list = [
    "xlm-roberta-base",
    "bert-base-multilingual-cased",
    "Davlan/bert-tiny-amharic"
]
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification
import numpy as np
import evaluate

# Define label mappings (use from Task 3)
id2label = {0: 'B-LOC', 1: 'B-PRICE', 2: 'B-Product', 3: 'I-LOC', 4: 'I-PRICE', 5: 'I-Product', 6: 'O'}
label2id = {v: k for k, v in id2label.items()}

# Metric
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

results = []

for model_name in model_list:
    print(f"\n🔁 Training model: {model_name}")
    
    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(label2id),
        id2label=id2label, label2id=label2id
    )

    # Tokenization
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

    tokenized_dataset = dataset.map(tokenize_and_align_labels)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=f"./{model_name.replace('/', '_')}_NER",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=1,
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
    eval_result = trainer.evaluate()
    
    results.append({
        "model": model_name,
        "f1": eval_result["eval_overall_f1"],
        "precision": eval_result["eval_overall_precision"],
        "recall": eval_result["eval_overall_recall"]
    })
# Sort by F1 score descending
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="f1", ascending=False)
print(results_df)
