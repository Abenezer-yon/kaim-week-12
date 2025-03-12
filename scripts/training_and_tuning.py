import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification
import numpy as np
from sklearn.metrics import classification_report
from scripts.tokenization_alignment import *
from scripts.load_conll_dataset import *

def set_training_arguments():
    return TrainingArguments(
        output_dir='./results',
        evaluation_strategy="no",  # Disable evaluation if no validation set
        logging_dir='./logs',
        per_device_train_batch_size=16,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=False,  # Disable loading best model if no validation
    )

def fine_tune_model(train_dataset, eval_dataset, model, tokenizer):
    if not train_dataset:
        raise ValueError("Training dataset is empty.")
    
    # Check for labels in the training dataset
    if 'labels' not in train_dataset or not train_dataset['labels']:
        raise ValueError("Labels cannot be empty in the training dataset.")

    # Convert to Dataset if it is not already
    if isinstance(train_dataset, dict):
        train_dataset = Dataset.from_dict(train_dataset)

    # Take only one example if needed
    if len(train_dataset) > 1:
        train_dataset = train_dataset.select([0])  # Only keep the first example

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=set_training_arguments(),
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_dataset else None,  # Allow for None
    )

    trainer.train()
    return trainer

def evaluate_model(trainer):
    """Evaluate the fine-tuned model."""
    eval_results = trainer.evaluate()
    return eval_results

def save_model(trainer, model_dir, tokenizer):
    """Save the fine-tuned model and tokenizer."""
    # Ensure model_dir is a string
    if not isinstance(model_dir, str):
        raise ValueError("model_dir must be a string representing the directory to save the model.")

    trainer.save_model(model_dir)  # Save the model to the specified directory
    tokenizer.save_pretrained(model_dir)  # Save the tokenizer to the same directory
    