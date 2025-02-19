import pandas as pd

# Load dataset
df = pd.read_csv("combined_dataset.csv")

# Function to count words in input_text
def word_count(text):
    return len(str(text).split())

# Categorize data based on word count
single_word_df = df[df["input_text"].apply(word_count) == 1]
two_to_three_df = df[df["input_text"].apply(word_count).between(2, 3)]
complex_phrases_df = df[df["input_text"].apply(word_count) > 3]

# Save to separate CSV files
single_word_df.to_csv("single_word.csv", index=False)
two_to_three_df.to_csv("two_to_three_words.csv", index=False)
complex_phrases_df.to_csv("complex_phrases.csv", index=False)

print("Datasets saved successfully!")
    