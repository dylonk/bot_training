from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the fine-tuned model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('./fine_tuned_t5_complex')
tokenizer = T5Tokenizer.from_pretrained('./fine_tuned_t5_complex')



test_cases = [
    ("house", "pluralize"),  # Expected: "angry bees buzzing what are probably your names"
]


for input_text, directive in test_cases:
    input_string = f"{directive}: {input_text}"
    inputs = tokenizer(input_string, return_tensors="pt", padding=True, truncation=True, max_length=64)
    outputs = model.generate(inputs['input_ids'])
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {input_text}")
    print(f"Directive: {directive}")
    print(f"Generated Output: {generated_text}\n")
