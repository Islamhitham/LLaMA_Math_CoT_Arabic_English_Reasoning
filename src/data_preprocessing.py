import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

# 1. Load dataset
dataset = load_dataset("miscovery/Math_CoT_Arabic_English_Reasoning", split="train")


# 2. Convert dataset to text pairs
def make_prompt_and_target(example):
    en_in = f"Language: English\nQuestion: {example['en_question']}\nAnswer:"
    ar_in = f"Language: Arabic\nQuestion: {example['ar_question']}\nAnswer:"

    return {
        "input_text": [en_in, ar_in],
        "target_text": [example["en_answer"], example["ar_answer"]],
    }

paired = dataset.map(make_prompt_and_target)
flat_dataset = paired.flatten()


# 3. Load tokenizer
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# 4. Format LLaMA Instruct example
def format_example(question, answer):
    return (
        "<|begin_of_text|>"
        "<|start_header|>user<|end_header|>\n"
        f"{question}\n"
        "<|start_header|>assistant<|end_header|>\n"
        f"{answer}"
    )


# 5. Tokenization with masking
def tokenize_fn(examples):
    texts = [format_example(q, a) for q, a in zip(examples["input_text"], examples["target_text"])]

    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=1024,
    )

    labels = []
    for q, a in zip(examples["input_text"], examples["target_text"]):
        full = format_example(q, a)
        full_ids = tokenizer(full, truncation=True, max_length=1024)["input_ids"]

        prompt = (
            "<|begin_of_text|>"
            "<|start_header|>user<|end_header|>\n"
            f"{q}\n"
            "<|start_header|>assistant<|end_header|>\n"
        )
        prompt_ids = tokenizer(prompt)["input_ids"]
        prompt_len = len(prompt_ids)

        lbl = full_ids.copy()
        lbl[:prompt_len] = [-100] * prompt_len

        if len(lbl) < 1024:
            lbl.extend([-100] * (1024 - len(lbl)))

        labels.append(lbl)

    tokenized["labels"] = labels
    return tokenized


print("Tokenizing...")
tokenized = flat_dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=[
        "input_text", "target_text",
        "en_question", "ar_question",
        "en_answer", "ar_answer",
        "category",
        "en_q_word", "ar_q_word",
        "en_a_word", "ar_a_word"
    ]
)

# 6. Train / Val / Test split
s1 = tokenized.train_test_split(test_size=0.2, seed=42)
s2 = s1["test"].train_test_split(test_size=0.5, seed=42)

final_dataset = DatasetDict({
    "train": s1["train"],
    "validation": s2["train"],
    "test": s2["test"]
})

final_dataset.save_to_disk("./processed_dataset")
print("Saved processed dataset to ./processed_dataset")