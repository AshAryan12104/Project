import json
import pandas as pd
from pathlib import Path


def process_topical_chat(input_dir, output_file):
    emotions = []
    texts = []
    
    # Process all JSON files in directory
    for file in Path(input_dir).glob('*.json'):
        with open(file) as f:
            data = json.load(f)
            for conv in data.values():
                for turn in conv['content']:
                    if 'emotion' in turn and 'message' in turn:
                        emotions.append(turn['emotion'])
                        texts.append(turn['message'])
    
    # Create DataFrame and save
    import os

# Create the parent directory if it does not exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df = pd.DataFrame({'text': texts, 'emotion': emotions})
    df.to_csv(output_file, index=False)
    

if __name__ == '__main__':
    process_topical_chat(
        input_dir=r'D:\gcetts\new project\hinglish-sentiment-analysis\data/raw/topical_chat',
        output_file=r'D:\gcetts\new project\hinglish-sentiment-analysis\data/processed/topical_chat_processed.csv'
    )
