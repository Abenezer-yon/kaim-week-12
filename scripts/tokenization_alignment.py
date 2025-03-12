
def tokenize_and_align_labels(sentences, labels, tokenizer, label_map):
    """Tokenize sentences and align labels with tokens."""
    # Tokenize without returning tensors
    tokenized_inputs = tokenizer(sentences, truncation=True, padding=True, is_split_into_words=True, return_tensors=None)

    labels_aligned = []
    
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)  # Special token
            else:
                # Ensure word_id is within bounds of the original label list
                if word_id < len(label):
                    label_ids.append(label_map[label[word_id]])  # Map label to ID
                else:
                    label_ids.append(-100)  # Fallback for out-of-bounds
        labels_aligned.append(label_ids)

    # Return structured output as lists
    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "token_type_ids": tokenized_inputs.get("token_type_ids", []),  # Handle cases without token_type_ids
        "labels": labels_aligned
    }