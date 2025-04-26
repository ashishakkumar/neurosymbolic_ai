import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import bitsandbytes as bnb
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import polars as pl
import json
from tqdm import tqdm


bnb_config = BitsAndBytesConfig(
  load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True)

model_name = '/home/ashish/llama_inference/openBIOLLM'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,device_map='cuda:1', quantization_config= bnb_config)

sys_prompt_other = """
You are an AI assistant analyzing clinical admission notes from the MIMIC IV dataset. Your task is to map symptoms mentioned in the notes to a list of provided diseases.

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
1. You will be provided with a list of diseases.
2. Carefully read the given clinical admission note.
3. Identify all symptoms as defined above mentioned in the note.
4. Map each symptom to relevant diseases from the provided list.
5. REMOVE any diseases that contain the word "unspecified" in their name from the mapped results.
6. Present your analysis in the following JSON format:

{
  "symptom1": ["disease1", "disease2", "disease7"],
  "symptom2": ["disease2", "disease3", "disease4"],
  "symptom3": ["disease1", "disease3", "disease5", "disease6"],
  ...
}

Important:
- Include a symptom ONLY if it meets the above definition, is explicitly mentioned in the note, AND maps to at least one disease from the provided list (after removing "unspecified" diseases).
- Completely exclude any symptoms that don't map to any remaining diseases in the provided list.
- Map each included symptom to ALL relevant diseases from the provided list, EXCEPT those containing "unspecified".
- A disease can be associated with multiple symptoms and thus can appear multiple times in the output.
- Do NOT include any diseases that don't have associated symptoms found in the note.
- Do NOT add any diseases or symptoms that are not explicitly stated in the input text or disease list.
- Keep disease names exactly as they appear in the provided list, without any modifications.
- COMPLETELY REMOVE any diseases that contain the word "unspecified" in their name from the final output.
- If removing "unspecified" diseases results in a symptom having no associated diseases, remove that symptom from the output as well.
- Focus on extracting and mapping symptoms concisely, prioritizing medically relevant details.

Now, analyze the following clinical admission note and map the symptoms to the provided list of diseases:
"""

prep_df = pl.read_csv('preprocessed.csv').to_pandas()
prep_df['long_title'] = prep_df['long_title'].apply(lambda x : x.split(','))

text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
failed_rows = []
limited_df = prep_df.head(2000)

def clear_vram():
    torch.cuda.empty_cache() 
    gc.collect()  


for i, j in tqdm(limited_df.iterrows(), total=limited_df.shape[0], desc="Processing rows"):
    notes = j.iloc[-1]
    long_title = j['long_title']

    messages = [
        {"role": "system", "content": f"{sys_prompt_other}"},
        {"role": "user", "content": f"### Clinical text :\n {notes} .### Diseases :\n {long_title}\n"},
    ]

    try:
        all_figures = text_generator(messages, max_new_tokens=500, num_return_sequences=1, temperature=0.1)[0]['generated_text'][-1]['content']
          

    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print("Caught OOM error, attempting to clear VRAM...")
            clear_vram()  
            continue 

    clear_vram()

    try:
        json_format = json.loads(all_figures)
        json_format['subject_id'] = j['subject_id']
        json_format['hadm_id'] = j['hadm_id']
        json_format['icd_10_exp'] = j['long_title']
        file_path = f"samples_symptoms/output_{j['subject_id']}_{j['hadm_id']}.json"

        with open(file_path, 'w') as file:
            json.dump(json_format, file, indent=4)
    
    except Exception as e:
        failed_rows.append((i, str(e)))

    torch.cuda.empty_cache()

if failed_rows:
    print("Failed to process the following rows:")
    for row in failed_rows:
        print(f"Row index: {row[0]}, Error: {row[1]}")
#nohup python neo4j_push.py > output.log 2>&1 &