import argparse

import settings
from transformers import TextStreamer
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=settings.output_model,
    max_seq_length=settings.max_seq_length,
    dtype=settings.dtype,
    load_in_4bit=settings.load_in_4bit,
)
# Enable native 2x faster inference
FastLanguageModel.for_inference(model)


def generate_text(prompt, max_new_tokens):
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    text_streamer = TextStreamer(tokenizer)

    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=max_new_tokens)


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with a pre-trained model"
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="Input prompt for text generation"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate",
    )

    args = parser.parse_args()
    generate_text(args.prompt, args.max_new_tokens)


if __name__ == "__main__":
    main()
