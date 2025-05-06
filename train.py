import pandas as pd
import torch
import numpy as np
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch import cuda
from torch.nn.functional import softmax


device = "cuda" if cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Dataset class
class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = str(self.dataframe.iloc[idx]['text'])
        label = int(self.dataframe.iloc[idx]['label'])  # Ensure int
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Load data
train_df = pd.read_csv(r'D:\gcetts\new project\hinglish-sentiment-analysis\data\processed\train.csv')
val_df = pd.read_csv(r'D:\gcetts\new project\hinglish-sentiment-analysis\data\processed\val.csv')

train_df = train_df.dropna(subset=['label'])
train_df['label'] = train_df['label'].astype(int)  # Convert float to int if needed
val_df = val_df.dropna(subset=['label'])
val_df['label'] = val_df['label'].astype(int)

# Verify correct number of classes
label_set = set(train_df['label'].unique())
assert label_set == {0, 1, 2, 3}, f"Expected labels {0,1,2,3} but got {label_set}"

# Tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

# Dataset & DataLoader
train_dataset = SentimentDataset(train_df, tokenizer, max_length=128)
val_dataset = SentimentDataset(val_df, tokenizer, max_length=128)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)

# Class weights
label_counts = train_df['label'].value_counts().sort_index()
num_classes = 4
assert len(label_counts) == num_classes, f"Expected {num_classes} class counts, got {len(label_counts)}"

class_weights = 1.0 / label_counts
class_weights = class_weights / class_weights.sum()
class_weights = torch.tensor(class_weights.values, dtype=torch.float).to(device)
print(f"Class weights: {class_weights}")

# Model
model = XLMRobertaForSequenceClassification.from_pretrained(
    'xlm-roberta-base',
    num_labels=num_classes
)
model = model.to(device)

# Optimizer and Loss Function
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# Early Stopping
best_val_acc = 0
epochs_no_improve = 0
patience = 2

# Training Loop
for epoch in range(10):
    model.train()
    total_loss = 0.0
    print(f"\nEpoch {epoch + 1}/10")
    print("-" * 30)

    for batch_idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_dataloader):
            print(f"  Batch {batch_idx + 1}/{len(train_dataloader)} - Loss: {loss.item():.4f}")

    avg_train_loss = total_loss / len(train_dataloader)

    # Validation
model.eval()
correct = 0
total = 0
all_preds = []
all_confidences = []

with torch.no_grad():
    for batch in val_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = softmax(outputs.logits, dim=1)  # Get probabilities
        confs, predicted = torch.max(probs, 1)  # Get top prediction + confidence

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_confidences.extend(confs.cpu().numpy())

val_acc = correct / total
print(f"Epoch {epoch + 1} completed - Avg Loss: {avg_train_loss:.4f} - Val Acc: {val_acc:.4f}")

# Optional: show predictions with confidence
for i in range(min(5, len(all_preds))):
    print(f"Prediction: {all_preds[i]} - Confidence: {all_confidences[i]:.4f}")

    # Check for early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        model.save_pretrained('models/trained_models/xlm-roberta-finetuned')
        tokenizer.save_pretrained('models/trained_models/xlm-roberta-finetuned')
        print("✅ Model improved and saved.")
    else:
        epochs_no_improve += 1
        print("No improvement.")
        if epochs_no_improve >= patience:
            print("🛑 Early stopping triggered.")
            break
