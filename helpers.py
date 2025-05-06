# Utility functions for the project

def load_model(model_path):
    """Loads the trained model"""
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model=model_path,
        tokenizer="xlm-roberta-base"
    )