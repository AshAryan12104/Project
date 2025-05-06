import pandas as pd
import numpy as np
import os
import warnings

# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

def combine_datasets():
    # File paths
    topical_file = r"D:\gcetts\new project\hinglish-sentiment-analysis\data\processed\topical_chat_processed.csv"
    hinglish_file = r"D:\gcetts\new project\hinglish-sentiment-analysis\data\processed\hinglish_processed.csv"

    # Check if files exist
    if not os.path.exists(topical_file):
        print(f"{topical_file} does not exist!")
        return
    
    if not os.path.exists(hinglish_file):
        print(f"{hinglish_file} does not exist!")
        return
    
    # Load processed datasets
    topical = pd.read_csv(topical_file)
    hinglish = pd.read_csv(hinglish_file)
    
    # Map emotions to consistent categories
    emotion_map = {
        'happy': 'positive',
        'excited': 'positive',
        'sad': 'negative',
        'angry': 'negative',
        'fearful': 'negative',
        'disgusted': 'negative',
        'neutral': 'neutral',
        'surprised': 'neutral'
    }
    
    topical['emotion'] = topical['emotion'].map(emotion_map)
    
    # Combine and shuffle
    combined = pd.concat([topical, hinglish])
    combined = combined.sample(frac=1).reset_index(drop=True)
    
    # Split into train/val/test (70/15/15)
    train, val, test = np.split(combined, [int(.7*len(combined)), int(.85*len(combined))])
    
    # Save final datasets
    output_dir = r"D:\gcetts\new project\hinglish-sentiment-analysis\data\processed"
    train.to_csv(f'{output_dir}/train.csv', index=False)
    val.to_csv(f'{output_dir}/val.csv', index=False)
    test.to_csv(f'{output_dir}/test.csv', index=False)

if __name__ == '__main__':
    combine_datasets()

