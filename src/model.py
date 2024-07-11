import settings
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=settings.model_name,
    max_seq_length=settings.max_seq_length,
    dtype=settings.dtype,
    load_in_4bit=settings.load_in_4bit,
)
model = FastLanguageModel.get_peft_model(
    model,
    target_modules=settings.target_modules,
    lora_alpha=32,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing=settings.use_gradient_checkpointing,
    random_state=3407,
    use_rslora=settings.use_rslora,
    loftq_config=settings.loftq_config,
)
EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
