from transformers import XLMRobertaForTokenClassification, DistilBertForTokenClassification, BertForTokenClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, Dataset

def fine_tune_model_multiple(model_names, train_dataset, eval_dataset):
    trainers = {}
    
    if eval_dataset is None:
        raise ValueError("No evaluation dataset provided. Ensure eval_dataset is not None.")
    
    if isinstance(train_dataset, dict):
        train_dataset = Dataset.from_dict(train_dataset)
    
    if isinstance(eval_dataset, dict):
        eval_dataset = Dataset.from_dict(eval_dataset)

    # Check if datasets are empty
    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        raise ValueError("Training or evaluation dataset is empty. Please provide valid datasets.")
    
    # Flatten the list of labels to identify unique labels
    all_labels = [label for sublist in train_dataset['labels'] for label in sublist if label != -100]
    unique_labels = set(all_labels)  # Create a set of unique labels
    num_labels = len(unique_labels)
    print(f"Unique labels: {unique_labels}, Number of labels: {num_labels}")

    # Ensure that labels are in the range [0, num_labels-1]
    label_map = {label: i for i, label in enumerate(sorted(unique_labels))}
    print(f"Label mapping: {label_map}")

    # Re-map labels in the dataset
    def remap_labels(labels):
        return [label_map.get(label, -100) for label in labels]

    train_dataset = train_dataset.map(lambda x: {'labels': remap_labels(x['labels'])})
    if eval_dataset:
        eval_dataset = eval_dataset.map(lambda x: {'labels': remap_labels(x['labels'])})

    for model_name in model_names:
        if model_name == "xlm-roberta":
            model = XLMRobertaForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=num_labels)
        elif model_name == "distilbert":
            model = DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)
        elif model_name == "mbert":
            model = BertForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=num_labels)
        else:
            raise ValueError(f"Model name '{model_name}' is not recognized.")
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=f"./{model_name}_output",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            logging_dir=f"./logs/{model_name}",
        )

        # Create Trainer instance
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Fine-tune the model
        trainer.train()
        
        trainers[model_name] = trainer
    
    return trainers