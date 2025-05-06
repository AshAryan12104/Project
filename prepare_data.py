import pandas as pd
import json
from pathlib import Path

def process_topical_chat():
    emotions = []
    texts = []
    
    for file in Path(r"D:\gcetts\new project\hinglish-sentiment-analysis\data/raw/topical_chat").glob("*.json"):
        with open(file) as f:
            data = json.load(f)
            for conv in data.values():
                for turn in conv['content']:
                    if 'emotion' in turn and 'message' in turn:
                        emotions.append(turn['emotion'])
                        texts.append(turn['message'])
    
    df = pd.DataFrame({'text': texts, 'emotion': emotions})
    df.to_csv(r"D:\gcetts\new project\hinglish-sentiment-analysis\data/processed/topical_chat_processed.csv", index=False)

def process_hinglish():
    # Sample Hinglish processing - adapt to your corpus
    samples = [
        ("Product accha hai", "positive"),
        ("Service bakwas thi", "negative"),
        ("Time par delivery hui", "positive"),
        ("Sab kuch kharab tha", "negative")
    ]
    df = pd.DataFrame(samples, columns=['text', 'emotion'])
    df.to_csv(r"D:\gcetts\new project\hinglish-sentiment-analysis\data/processed/hinglish_processed.csv", index=False)

if __name__ == "__main__":
    process_topical_chat()
    process_hinglish()