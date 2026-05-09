from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    # Load the Qwen3 small model and corresponding tokenizer
    model_name = "Qwen/Qwen3-0.6B"  # Small model version of Qwen3

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    
    # Set up the conversation prompt
    prompt = "Hello, please introduce yourself."

    # Generate a response
    print("\nUser input:", prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nModel response:", response)

if __name__ == "__main__":
    main()