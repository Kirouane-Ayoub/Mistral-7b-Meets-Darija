from datasets import load_dataset
from utils import arabic_formatting_prompts_func, french_formatting_prompts_func

# Define constants for dataset names and splits
DATASETS = {
    "arabic": {
        "name": "ayoubkirouane/Arabic_mix",
        "split": "train",
        "formatting_func": arabic_formatting_prompts_func,
    },
    "french": {
        "name": "ayoubkirouane/French_mix",
        "split": "train",
        "formatting_func": french_formatting_prompts_func,
    },
    "darija": {
        "name": "ayoubkirouane/Algerian-Darija",
        "split": "v1",
        "formatting_func": None,  # No formatting function for Darija
    },
}


def load_and_process_dataset(dataset_info):
    """Load and process a dataset based on provided name and dataset_info."""
    dataset = load_dataset(dataset_info["name"], split=dataset_info["split"])
    if dataset_info["formatting_func"]:
        dataset = dataset.map(dataset_info["formatting_func"], batched=True)
    return dataset
