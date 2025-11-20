import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftConfig, get_peft_model



MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_PATH = "../llama_mathcot_qlora_adapter"

# 4-bit quantization config 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

print("Loading LoRA adapter...")
peft_cfg = PeftConfig.from_pretrained(ADAPTER_PATH)
model = get_peft_model(base_model, peft_cfg)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token



def generate_answer(question: str, lang: str = "English"):
    prompt = f"Language: {lang}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.2,
            top_p=0.9,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)



if __name__ == "__main__":
    q1 = "If 4 apples cost $2, how much do 10 apples cost?"
    print("English Example:")
    print(generate_answer(q1, "English"), "\n")

    q2 = "إذا كانت تكلفة 4 تفاحات 2 دولار، فما هي تكلفة 10 تفاحات؟"
    print("Arabic Example:")
    print(generate_answer(q2, "Arabic"))
