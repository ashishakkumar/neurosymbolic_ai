

from unsloth import FastLanguageModel
import torch
max_seq_length = 8192 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = ""YOUR_HF_TOKEN"...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

from datasets import load_dataset
dataset = load_dataset("ashishkgpian/symptoms2diseases_another_way", split="train")

def output_col(row):
    # Convert the list to a string directly
    return {'llm_output': str(row['icd-codes'])}

dataset = dataset.map(output_col)


alpaca_prompt = """
{}

### Clinical text :
{} 

### Response:
{}
"""

instruction = """
You are an AI assistant analyzing clinical admission notes from the MIMIC IV dataset. Your task is to map symptoms mentioned in the notes to a list of ICD-9 diseases codes.

Definition of a Symptom:
A symptom is a physical or mental feature which is regarded as indicating a condition of disease, particularly such a feature that is apparent to the patient. Focus on current, active symptoms experienced by the patient.

What NOT to Include as Symptoms:
- Medical history (e.g., "history of drug use", "smoking history")
- Family history
- Test results or medical observations (e.g., "hypoechoic lesion")
- Diagnoses or named conditions
- Risk factors
- Treatments or medications

Instructions:

1. Carefully read the given clinical admission note.
2. Identify all symptoms as defined above mentioned in the note.
3. Map each symptom to relevant ICD-9 diseases codes from the provided list.
4. Present your analysis in the following list format:

['icd_9_code_1', 'icd_9_code_2', 'icd_9_code_3', 'icd_9_code_4', ... ] 

Important:
- Completely exclude any symptoms that don't map to any icd-9 diseases code.
- Map each included symptom to ALL relevant icd-9 diseases codes from the provided list, EXCEPT those containing "unspecified".
- An icd-9 disease code can be associated with multiple symptoms and thus can appear multiple times in the output.
- Do NOT include any icd-9 disease codse that don't have associated symptoms found in the note.

Now, analyze the following clinical admission note and map the symptoms to the provided list of diseases codes:"""


EOS_TOKEN = tokenizer.eos_token  

def formatting_prompts_func(examples):
    inputs = examples["note"]
    outputs = examples["llm_output"]
    texts = []
    for input, output in zip(inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"prompt": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

import gc
torch.cuda.empty_cache()
gc.collect()

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "prompt",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        # max_steps = 60,
        num_train_epochs = 50, # For longer training runs!
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer_stats = trainer.train()
model.save_pretrained("Biollama_icd9_codes") # Local saving
tokenizer.save_pretrained("Biollama_icd9_codes")
model.save_pretrained_merged("Biollama_icd9_codes_merged", tokenizer, save_method = "merged_4bit_forced",)
model.push_to_hub_merged("ashishkgpian/Biollama_icd9_codes", tokenizer, save_method = "merged_4bit_forced", token = ""YOUR_HF_TOKEN"")
torch.cuda.empty_cache()
gc.collect()


