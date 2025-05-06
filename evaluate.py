from datasets import load_dataset
from sklearn.metrics import classification_report
from transformers import pipeline

LABEL_MAP = {
    0: "happy",
    1: "angry",
    2: "sad", 
    3: "neutral"
}

def evaluate():
    # Load test data
    dataset = load_dataset('csv', data_files={'test': 'data/processed/test.csv'})['test']
    
    # Load model and tokenizer
    classifier = pipeline(
        "text-classification",
        model="models/trained_models/xlm-roberta-finetuned",
        tokenizer="models/trained_models/xlm-roberta-finetuned"
    )

    true_labels = []
    pred_labels = []

    for row in dataset:
        result = classifier(row['text'])[0]
        pred_id = int(result['label'].split('_')[-1])
        pred_labels.append(pred_id)
        true_labels.append({
            'happy': 0, 'angry': 1, 'sad': 2, 'neutral': 3
        }[row['emotion']])
    
    print(classification_report(true_labels, pred_labels, target_names=LABEL_MAP.values()))

if __name__ == '__main__':
    evaluate()
