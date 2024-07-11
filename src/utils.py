import re

import torch
from model import EOS_TOKEN

# Constants for prompts
ARABIC_PROMPT_TEMPLATE = """
العنوان: {}
النص :
{}
"""

FRENCH_PROMPT_TEMPLATE = """
titre : {}
text :
{}
"""


def clean_newlines(text):
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


def format_prompts(examples, prompt_template):
    return {
        "text": [
            prompt_template.format(title, clean_newlines(text)) + EOS_TOKEN
            for title, text in zip(examples["title"], examples["text"])
        ]
    }


def arabic_formatting_prompts_func(examples):
    return format_prompts(examples, ARABIC_PROMPT_TEMPLATE)


def french_formatting_prompts_func(examples):
    return format_prompts(examples, FRENCH_PROMPT_TEMPLATE)


def get_current_gpu_stats(gpu_stats):
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    return start_gpu_memory, max_memory


def get_final_gpu_stats(start_gpu_memory, max_memory):
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    return used_memory, used_memory_for_lora, used_percentage, lora_percentage
