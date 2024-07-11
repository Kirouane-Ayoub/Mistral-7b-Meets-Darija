import settings
import torch
from datasets import DATASETS, load_and_process_dataset
from model import model, tokenizer
from unsloth import UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported
from utils import get_current_gpu_stats, get_final_gpu_stats


def create_trainer(
    train_dataset,
    learning_rate,
    embedding_learning_rate,
    lr_scheduler_type,
    dataset_text_field="text",
    num_train_epochs=10,
):
    return UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field=dataset_text_field,
        max_seq_length=settings.max_seq_length,
        dataset_num_proc=8,
        args=UnslothTrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_ratio=0.1,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            embedding_learning_rate=embedding_learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim=settings.optimizer,
            weight_decay=0.00,
            lr_scheduler_type=lr_scheduler_type,
            seed=3407,
            output_dir="outputs",
        ),
    )


gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory, max_memory = get_current_gpu_stats(gpu_stats)

print("-------- Current memory stats --------")
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


def train_dataset(trainer, round_name):
    print(f"Start {round_name} Round of Training....")
    trainer_stats = trainer.train()
    return trainer_stats


arabic = load_and_process_dataset("arabic", DATASETS["arabic"])
trainer = create_trainer(arabic, 5e-5, 5e-6, "cosine")
arabic_stats = train_dataset(trainer, "The First")

french = load_and_process_dataset("french", DATASETS["french"])
trainer = create_trainer(french, 5e-5, 1e-5, "linear")
french_stats = train_dataset(trainer, "The Second")

darija = load_and_process_dataset("darija", DATASETS["darija"])
trainer = create_trainer(darija, 5e-5, 1e-5, "linear", "Text", 20)
darija_stats = train_dataset(trainer, "The Final")

print("Saving Lora pretrained model with the tokenizer...")
model.save_pretrained(settings.output_model)
tokenizer.save_pretrained(settings.output_model)

print("-------- Final memory stats --------")
(
    used_memory,
    used_memory_for_lora,
    used_percentage,
    lora_percentage,
) = get_final_gpu_stats(start_gpu_memory, max_memory)
print(f"{darija_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(darija_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# Merge to 16bit
print("Save pretrained merges model on 16 bit (llama-cpp) .....")
model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")

# Empty Cache
torch.cuda.empty_cache()
