import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV data
df = pd.read_csv(r"D:\gcetts\new project\hinglish-sentiment-analysis\data\processed\Text,Sentiment,Label.csv")

# Display the first few rows to ensure the data is loaded correctly
print(df.head())

# Split the data into 80% train and 20% validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the split data into CSV files
train_df.to_csv(r'D:\gcetts\new project\hinglish-sentiment-analysis\data\processed\train.csv', index=False)
val_df.to_csv(r'D:\gcetts\new project\hinglish-sentiment-analysis\data\processed\val.csv', index=False)

print("Data has been split and saved as train.csv and val.csv")
