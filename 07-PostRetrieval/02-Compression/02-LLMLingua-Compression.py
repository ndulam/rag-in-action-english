"""
LLMLingua Text Compression Example
LLMLingua is a tool for compressing long texts, capable of reducing token count while preserving core information.
Suitable for post-retrieval processing in RAG systems to reduce LLM input costs.
"""

from llmlingua import PromptCompressor

# =============================================================================
# 1. Initialize the PromptCompressor
# =============================================================================

# Force use of CPU device to avoid CUDA errors
# model_name: specifies the base model used for compression, here using Llama-2-7b
# device_map: device mapping, "cpu" means using CPU to run, avoiding GPU memory issues
llm_lingua = PromptCompressor(
    model_name="NousResearch/Llama-2-7b-hf",
    device_map="cpu"  # Explicitly specify CPU usage
)

# =============================================================================
# 2. Text compression example
# =============================================================================

# Compress long text content
# context: the original text content to be compressed
# instruction: compression instruction, tells the model how to compress
# question: related question (optional), helps the model understand the compression focus
# target_token: target token count, the compressed text should be within this length
compressed_prompt = llm_lingua.compress_prompt(
    context="Yungang Grottoes is located 17 kilometers west of Datong City, Shanxi Province, northern China, on the southern foot of Wuzhou Mountain. The grottoes are carved into the mountain and extend 1 kilometer from east to west. There are 45 main caves, 252 large and small niches, and over 51,000 stone sculptures, making it one of the largest ancient grotto groups in China. Along with Dunhuang Mogao Caves, Luoyang Longmen Grottoes, and Tianshui Maijishan Grottoes, it is known as one of China's four great grotto art treasures. In 1961, it was announced by the State Council as one of the first batch of key national cultural relics protection units. On December 14, 2001, it was inscribed on the UNESCO World Heritage List. On May 8, 2007, it was rated as one of the first national 5A-level tourist attractions by the National Tourism Administration...",
    instruction="Compress and retain the main content",
    question="",
    target_token=10  # Set target token count
)

# Output the compressed text
print("=== Compressed Text ===")
print(compressed_prompt)
print(compressed_prompt['compressed_prompt'])
print(f"Compression ratio: {compressed_prompt.get('rate', 'N/A')}")

# =============================================================================
# 3. JSON data compression example
# =============================================================================

# Define the JSON data to be compressed
json_data = {
    "id": 987654,
    "name": "Wukong",
    "biography": "Sun Wukong, also known as the Monkey King, is one of the main characters in the classic Chinese fantasy novel Journey to the West. Sun Wukong was born from a magical stone that had been nurtured since the creation of the world. He learned the 72 Earthly Transformations from Master Puti, obtained the Ruyi Jingu Bang (golden staff) from the Dragon Palace as his weapon, and caused great havoc in Heaven. He was then suppressed under Five Elements Mountain by the Buddha and could not move. Five hundred years later, the Tang Monk passed by Five Elements Mountain on his journey to the West to obtain scriptures, removed the seal, and rescued Sun Wukong. Sun Wukong, filled with gratitude, was guided by Guanyin Bodhisattva and became a disciple of the Tang Monk, journeying together to the West to obtain scriptures."
}

# JSON compression configuration
# rate: compression ratio, between 0-1, smaller means more compression
# compress: whether to compress this field
# pair_remove: whether to allow removal of key-value pairs
# value_type: type of value (number/string, etc.)
json_config = {
    "id": {"rate": 1, "compress": False, "pair_remove": False, "value_type": "number"},      # ID not compressed, kept as-is
    "name": {"rate": 0.7, "compress": False, "pair_remove": False, "value_type": "string"}, # Name lightly compressed
    "biography": {"rate": 0.3, "compress": True, "pair_remove": False, "value_type": "string"}  # Biography heavily compressed
}

# Execute JSON compression - with exception handling
try:
    print("\n=== Original JSON Data ===")
    import json
    print(json.dumps(json_data, ensure_ascii=False, indent=2))

    compressed_json = llm_lingua.compress_json(json_data, json_config)

    # Output the compressed JSON
    print("\n=== Compressed JSON ===")
    print(compressed_json['compressed_prompt'])

except Exception as e:
    print(f"\n=== JSON Compression Error ===")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("Possible solutions:")
    print("1. Simplify the text content in JSON")
    print("2. Avoid using special characters and escape characters")
    print("3. Adjust compression parameters")

    # Try fallback: compress only the text content
    print("\n=== Using Fallback Text Compression Method ===")
    try:
        backup_compressed = llm_lingua.compress_prompt(
            context=json_data["biography"],
            instruction="Compress the biography content, retain key information",
            question="",
            target_token=30
        )
        print("Compressed biography:")
        print(backup_compressed['compressed_prompt'])
    except Exception as backup_e:
        print(f"Fallback method also failed: {backup_e}")

# =============================================================================
# Usage Instructions
# =============================================================================
"""
Main application scenarios for the compressor:
1. Compress retrieved documents in RAG systems to reduce input tokens
2. Compress conversation history to maintain context coherence
3. Compress structured data (JSON) while preserving key fields

Parameter tuning recommendations:
- target_token: set according to the context limit of the downstream LLM
- rate: adjust according to information importance, set higher for key information
- compress: numeric data usually does not need compression
"""
