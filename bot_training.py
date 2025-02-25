from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset,Dataset, DatasetDict
import pandas as pd
import sentencepiece
import evaluate
from sklearn.model_selection import train_test_split


print(sentencepiece.__version__)



# Load pre-trained T5-small model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('fine_tuned_t5_complex')
tokenizer = T5Tokenizer.from_pretrained('fine_tuned_t5_complex')
# Step 1: Load CSV into a pandas DataFrame and drop unnecessary column
df = pd.read_csv('special_edge_cases.csv')    
df = df.drop(columns=['__index_level_0__'], errors='ignore')
# df is type: pandas.DataFrame
    
metric = evaluate.load("accuracy")


print(f"DataFrame columns before processing: {df.columns}")


def prepare_datasets(df):
    # Clean the data
    df = df[['directive', 'input_text', 'output_text']]  # Only keep relevant columns
    
    # Convert all columns to string and handle NaN values
    df['directive'] = df['directive'].fillna('').astype(str)
    df['input_text'] = df['input_text'].fillna('').astype(str)
    df['output_text'] = df['output_text'].fillna('').astype(str)
    
    # Remove any empty strings
    df = df[df['output_text'].str.len() > 0]
    
    train_df, eval_df = train_test_split(df, test_size=0.1)
    
    return DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'eval': Dataset.from_pandas(eval_df)
    })
# datasets becomes type: datasets.DatasetDict
datasets = prepare_datasets(df)
# Usage would look like:

# Access the train dataset from your DatasetDict
# train_dataset becomes type: datasets.Dataset
train_dataset = datasets['train']
eval_dataset = datasets['eval']

# Tokenization function
def tokenize_function(examples):
    # examples is a dict with keys 'directive', 'input_text', 'output_text'
    # each key contains a list of values
    
    inputs = [f"{d}: {i}" for d, i in zip(examples['directive'], examples['input_text'])]
    
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=64)
    labels = tokenizer(examples['output_text'], padding="max_length", truncation=True, max_length=64)
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# This line transforms the Dataset
# Apply tokenization to the datasets
train_dataset = datasets['train'].map(tokenize_function, batched=True, remove_columns=['directive', 'input_text', 'output_text', '__index_level_0__'])
eval_dataset = datasets['eval'].map(tokenize_function, batched=True, remove_columns=['directive', 'input_text', 'output_text', '__index_level_0__'])
# train_dataset is still type: datasets.Dataset but now contains tokenized data

# Debugging: Print the dataset columns after tokenization
print(f"Train dataset columns after tokenization: {train_dataset.column_names}")
print(f"Eval dataset columns after tokenization: {eval_dataset.column_names}")


def compute_metrics(pred):
    predictions, references = pred.predictions, pred.label_ids
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_refs = tokenizer.batch_decode(references, skip_special_tokens=True)
    results = metric.compute(predictions=decoded_preds, references=decoded_refs)
    print(f"Evaluation Results: {results}")

    return results

# Set up the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate = 1e-5, 
    per_device_train_batch_size=8,  # Can be much higher with shorter sequences
    num_train_epochs=2,  # Might converge faster
    warmup_steps=10,
    lr_scheduler_type='cosine',
    gradient_accumulation_steps=1,
    eval_steps=100,  # More frequent evaluation
    remove_unused_columns=False,  # Add this line
    logging_dir='./logs',  # Save logs here
    logging_steps=50,      # Log every 50 steps
    save_steps=100,  
    save_strategy='steps',  # Save after every 'save_steps' steps

)

# Initialize the Trainer., Trainer will both train and evaluate
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Pass the tokenized train dataset
    eval_dataset=eval_dataset,    # Pass the tokenized eval dataset
    compute_metrics=compute_metrics,  # Make sure this line is included

)

# Start training
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained('./fine_tuned_t5_complex_test')
tokenizer.save_pretrained('./fine_tuned_t5_complex_test')