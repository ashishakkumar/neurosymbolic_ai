

import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer
from transformers import EarlyStoppingCallback
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset, DatasetDict
from unsloth import FastLanguageModel
import torch
import gc
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

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
    "unsloth/Phi-3-mini-4k-instruct",          # Phi-3 2x faster!d
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "microsoft/Phi-3-mini-4k-instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = ""YOUR_HF_TOKEN"", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
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



train_data =  load_dataset('csv', data_files='/home/ashish/llama_inference/clinical-outcome-prediction/saved_files/DIA_GROUPS_3_DIGITS_adm_train.csv')
val_data = load_dataset('csv', data_files='/home/ashish/llama_inference/clinical-outcome-prediction/saved_files/DIA_GROUPS_3_DIGITS_adm_val.csv')

dataset = DatasetDict({
    'train': train_data['train'], 
    'val': val_data['train']     
})

train_dataset = dataset['train']
val_dataset = dataset['val']

def output_col(row):
    dicts_  = {row['long_texts']}
    return {'llm_output':str(dicts_)}

train_dataset = train_dataset.map(output_col)
val_dataset = val_dataset.map(output_col)

alpaca_prompt = """
{}

### Clinical text :
{} 

### Response:
{}
"""

instruction = """
You are an AI assistant analyzing clinical admission notes from the MIMIC III dataset. Your task is to understand the symptoms mentioned in the notes and then output the long_text format of the diseases mentioned in ICD-9 dataset which are relevant to the symptoms mentioned in the notes.

"""


EOS_TOKEN = tokenizer.eos_token  

def formatting_prompts_func(examples):
    inputs = examples["text"]
    outputs = examples["llm_output"]
    texts = []
    for input, output in zip(inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"prompt": texts}

train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
val_dataset = val_dataset.map(formatting_prompts_func, batched=True)

# trainer = SFTTrainer(
#     model = model,
#     tokenizer = tokenizer,
#     train_dataset = train_dataset,
#     eval_dataset = val_dataset,
#     dataset_text_field = "prompt",
#     max_seq_length = max_seq_length,
#     dataset_num_proc = 2,
#     packing = False, # Can make training 5x faster for short sequences.
#     args = TrainingArguments(
#         per_device_train_batch_size = 20,
#         gradient_accumulation_steps = 8,
#         warmup_steps = 5,
#         # max_steps = 1,
#         num_train_epochs = 50, # For longer training runs!
#         learning_rate = 2e-4,
#         fp16 = not is_bfloat16_supported(),
#         bf16 = is_bfloat16_supported(),
#         logging_steps = 1,
#         optim = "adamw_8bit",
#         weight_decay = 0.01,
#         lr_scheduler_type = "linear",
#         seed = 3407,
#         output_dir = "outputs",
#     ),
# )


from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    # eval_dataset = val_dataset,
    dataset_text_field = "prompt",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 7,
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        # max_steps = 1,
        num_train_epochs = 2, # For longer training runs!
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

# args = TrainingArguments(
#     per_device_train_batch_size=10,
#     gradient_accumulation_steps=5,
#     warmup_steps=5,
#     max_steps=1,
#     # num_train_epochs=1,
#     learning_rate=2e-4,
#     fp16=not is_bfloat16_supported(),
#     bf16=is_bfloat16_supported(),
#     logging_steps=1,
#     optim="adamw_8bit",
#     weight_decay=0.01,
#     lr_scheduler_type="linear",
#     seed=3407,
#     output_dir="phi_outputs",
#     load_best_model_at_end=True,
#     evaluation_strategy='epoch',
#     eval_steps=1,
#     metric_for_best_model='mse',
#     save_strategy='epoch',  # Save the model after each epoch
# )

# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=train_dataset,
#     dataset_text_field="prompt",
#     max_seq_length=max_seq_length,
#     packing=False,
#     args=args,

# )



torch.cuda.empty_cache()
gc.collect()


trainer_stats = trainer.train()
model.save_pretrained("llama_texts") # Local saving
tokenizer.save_pretrained("llama_texts")
model.save_pretrained_merged("llama_texts_merged", tokenizer, save_method = "merged_4bit_forced",)
model.push_to_hub_merged("ashishkgpian/BioLlama_texts", tokenizer, save_method = "merged_4bit_forced", token = ""YOUR_HF_TOKEN"")
torch.cuda.empty_cache()
gc.collect()