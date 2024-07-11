max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally .
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.
use_rslora = (True,)  # We support rank stabilized LoRA
loftq_config = (None,)  # And LoftQ
use_gradient_checkpointing = "unsloth"  # True or "unsloth" for very long context
target_modules = (
    [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "embed_tokens",
        "lm_head",
    ],
)  # Add for continual pretraining

output_model = "Mistral-Darija"
optimizer = "adamw_8bit"

"""
The Mistral-7B-v0.3 Large Language Model (LLM) is a Mistral-7B-v0.2 with extended vocabulary.
"""
model_name = "mistralai/Mistral-7B-v0.3"
