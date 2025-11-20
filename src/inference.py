import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftConfig, get_peft_model

model_name = "meta-llama/Llama-3.2-3B-Instruct"
adapter_path = "./llama_mathcot_qlora_adapter"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load base + adapter
base = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

config = PeftConfig.from_pretrained(adapter_path)
model = get_peft_model(base, config)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_answer(question, lang="English"):
    prompt = f"Language: {lang}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example:
print(generate_answer("If 4 apples cost $2, how much do 10 apples cost?", "English"))
print(generate_answer("إذا كانت تكلفة 4 تفاحات 2 دولار، فما هي تكلفة 10 تفاحات؟", "Arabic"))