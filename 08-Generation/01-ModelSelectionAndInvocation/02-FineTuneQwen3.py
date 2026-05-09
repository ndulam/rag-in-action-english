from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import torch
import os

def prepare_dataset(tokenizer):
    # Load a test dataset (using a simple Q&A dataset as an example)
    dataset = load_dataset("squad", split="train[:100]")  # Use first 100 samples as example
    
    def format_prompt(example):
        # Format the SQuAD dataset into a conversational format
        prompt = f"Question: {example['question']}\nContext: {example['context']}\nAnswer: {example['answers']['text'][0]}"
        return {"text": prompt}
    
    # Format the dataset
    dataset = dataset.map(format_prompt)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def main():
    # Set model name and output directory
    model_name = "Qwen/Qwen3-0.6B"
    output_dir = "./qwen3_finetuned"
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare dataset
    print("Preparing dataset...")
    dataset = prepare_dataset(tokenizer)
    
    # Set training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # Number of training epochs
        per_device_train_batch_size=4,  # Batch size per device
        gradient_accumulation_steps=4,  # Gradient accumulation steps
        learning_rate=2e-5,  # Learning rate
        weight_decay=0.01,  # Weight decay
        warmup_steps=100,  # Warmup steps
        logging_steps=10,  # Logging steps
        save_steps=100,  # Steps between checkpoint saves
        fp16=True,  # Use mixed precision training
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Do not use masked language modeling
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save model
    print(f"Training complete. Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Test the fine-tuned model
    print("\nTesting the fine-tuned model...")
    test_prompt = "Question: What is artificial intelligence?\nAnswer:"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nModel response:", response)

if __name__ == "__main__":
    main()