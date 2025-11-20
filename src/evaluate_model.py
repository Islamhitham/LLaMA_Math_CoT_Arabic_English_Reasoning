import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftConfig, get_peft_model
import math

dataset = load_from_disk("./processed_dataset")
model_name = "meta-llama/Llama-3.2-3B-Instruct"
adapter_path = "./llama_mathcot_qlora_adapter"

bnb_config = BitsAndBytesConfig(load_in_4bit=True)

base = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

config = PeftConfig.from_pretrained(adapter_path)
model = get_peft_model(base, config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def evaluate_split(split="test"):
    total_loss = 0
    count = 0

    for batch in dataset[split]:
        ids = torch.tensor(batch["input_ids"]).unsqueeze(0).to(model.device)
        mask = torch.tensor(batch["attention_mask"]).unsqueeze(0).to(model.device)
        labels = torch.tensor(batch["labels"]).unsqueeze(0).to(model.device)

        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask, labels=labels)

        total_loss += out.loss.item()
        count += 1

    avg = total_loss / count
    ppl = math.exp(avg)
    print(f"[{split}] avg loss = {avg:.4f}, perplexity = {ppl:.2f}")

evaluate_split("test")
