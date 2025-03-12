import pandas as pd

def read_subset_df(data):
    # Load the dataset
    df = pd.read_csv(data)  # Load the preprocessed data
    print(df.head())  # Inspect the first few rows
    print(df.columns)  # Print the column names

    # Trim spaces from column names
    df.columns = df.columns.str.strip()

    # Check if 'content' column exists
    if 'content' not in df.columns:
        raise KeyError("The column 'content' does not exist in the DataFrame.")

    # Drop rows with missing values in 'content'
    df = df.dropna(subset=['content'])

    # Select a subset of messages (30-50)
    subset_df = df['content'].sample(n=50, random_state=1)  # Randomly select 50 messages

    return subset_df

# Function to label entities
def label_entities(message):
    tokens_labels = []
    tokens = message.split()  # Simple tokenization

    for token in tokens:
        # Placeholder logic for labeling
        if token.lower() in ["baby", "bottle"]:
            label = "B-Product" if token == "Baby" else "I-Product"
        elif token.lower() in ["addis", "abeba", "bole"]:
            label = "B-LOC" if token == "Addis" else "I-LOC"
        elif "ዋጋ" in token or "ብር" in token:
            label = "B-PRICE" if token.startswith("ዋጋ") else "I-PRICE"
        else:
            label = "O"
        
        tokens_labels.append(f"{token}\t{label}")

    return "\n".join(tokens_labels)

def label_data(subset_df):
    # Create CoNLL formatted strings
    conll_data = []
    for message in subset_df:
        labeled_message = label_entities(message)
        conll_data.append(labeled_message + "\n")  # Blank line between messages

    # Save to a plain text file
    with open('labeled_data.conll', 'w', encoding='utf-8') as f:
        f.writelines(conll_data)

    print("Labeled data saved to labeled_data.conll")
