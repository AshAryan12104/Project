from transformers import XLMRobertaTokenizer

class HinglishTokenizer:
    def __init__(self):
        self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        self.add_hinglish_tokens()
    
    def add_hinglish_tokens(self):
        hinglish_words = ["accha", "bakwas", "thoda", "hoon"]
        self.tokenizer.add_tokens(hinglish_words)
    
    def tokenize(self, text):
        return self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)