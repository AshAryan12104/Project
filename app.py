from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model and tokenizer (adjust path if needed)
model_name = r"models/trained_models/xlm-roberta-finetuned"  # replace with your actual model path or huggingface ID
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Your label mapping
id2label = {
    0: "happy",
    1: "sad",
    2: "angry",
    3: "neutral"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')

    if not text.strip():
        return jsonify({'sentiment': 'neutral'})

    # Tokenize and predict
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        sentiment = id2label[pred_id]

    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
