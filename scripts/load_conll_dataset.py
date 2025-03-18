
def load_conll_dataset(file_path):
    sentences = []
    labels = []
    current_sentence = []
    current_labels = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            print(f"Reading line: {line.strip()}")  # Debugging line
            if line.strip():
                try:
                    token, label = line.split()
                    current_sentence.append(token)
                    current_labels.append(label)
                except ValueError:
                    print(f"Skipping line due to ValueError: {line.strip()}")
            else:
                if current_sentence:
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence = []
                    current_labels = []

    # Handle the last sentence if the file does not end with a blank line
    if current_sentence:
        sentences.append(current_sentence)
        labels.append(current_labels)

    return sentences, labels