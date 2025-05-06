import pandas as pd
import re
import os

def clean_hinglish(text):
    # Remove special characters but preserve Hinglish words
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text.lower()

def process_hinglish(input_file, output_file):
    # Check if the input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"The input file does not exist: {input_file}")

    df = pd.read_csv(input_file)
    df['text'] = df['text'].apply(clean_hinglish)
    
    # Ensure the directory exists before saving the file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df.to_csv(output_file, index=False)

if __name__ == '__main__':
    process_hinglish(
        input_file = r'D:\gcetts\new project\hinglish-sentiment-analysis\data\raw\hinglish_corpus\hinglish_samples.csv'
,
        output_file=r'D:\gcetts\new project\hinglish-sentiment-analysis\data/processed/hinglish_processed.csv')
