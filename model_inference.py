import torch
import torch.nn.functional as F
from transformers import XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification
import os

# Load model and tokenizer
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'trained_models', 'xlm-roberta-finetuned'))
tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_path)
model = XLMRobertaForSequenceClassification.from_pretrained(model_path)

# Ensure model is in eval mode
model.eval()

# Make predictions
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).squeeze()

    confidence, predicted_class_id = torch.max(probs, dim=0)
    label = model.config.id2label[str(predicted_class_id.item())]
    confidence_score = round(confidence.item(), 4)

    return {"label": label, "confidence": confidence_score}

