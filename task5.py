from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

model_path = "finetuned_amharic_ner"  # Your model directory
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
sample = "በአዲስ አበባ ዋጋ 1000 ብር የህፃናት ጫማ አለ"
preds = ner_pipeline(sample)
for ent in preds:
    print(f"{ent['word']} → {ent['entity_group']} ({ent['score']:.2f})")
from lime.lime_text import LimeTextExplainer

# Define a wrapper for LIME
class NERPredictWrapper:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.labels = ["B-PRICE", "I-PRICE", "B-LOC", "I-LOC", "B-Product", "I-Product", "O"]
    
    def predict_proba(self, texts):
        results = []
        for text in texts:
            preds = self.pipeline(text)
            # Build one-hot labels (just for simplicity)
            probs = [0] * len(self.labels)
            for pred in preds:
                idx = self.labels.index(pred["entity_group"])
                probs[idx] = pred["score"]
            results.append(probs)
        return results

explainer = LimeTextExplainer(class_names=["B-PRICE", "I-PRICE", "B-LOC", "I-LOC", "B-Product", "I-Product", "O"])

wrapper = NERPredictWrapper(ner_pipeline)
exp = explainer.explain_instance(sample, wrapper.predict_proba, num_features=10)

exp.show_in_notebook()
import shap
import torch

# Use the tokenizer and model
def predict_shap(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, is_split_into_words=False)
    with torch.no_grad():
        outputs = model(**inputs).logits
    probs = torch.nn.functional.softmax(outputs, dim=-1)
    return probs[:, :, model.config.label2id["B-Product"]].numpy()

# Create SHAP explainer
explainer = shap.Explainer(predict_shap, tokenizer)
shap_values = explainer([sample])

# Visualize
shap.plots.text(shap_values[0])
examples = [
    "በአዲስ አበባ ዋጋ 1000 ብር",  # location + price
    "1000 ብር ዋጋ የህፃናት ሻምፑ",  # price + product
    "በቦሌ በነፃ የተሰጠ መጽሐፍ",  # overlapping entity (free price + location)
]

for text in examples:
    print(f"\n Text: {text}")
    preds = ner_pipeline(text)
    for ent in preds:
        print(f"→ {ent['word']} = {ent['entity_group']} ({ent['score']:.2f})")
