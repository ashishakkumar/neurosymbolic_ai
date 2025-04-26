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

model_name = 'aaditya/Llama3-OpenBioLLM-70B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,device_map='cuda:1', quantization_config = bnb_config)


sys_prompt = """
You are an AI assistant analyzing clinical discharge summary notes from the MIMIC III dataset. Focus on these key sections:

1. Symptoms: Current physical or mental signs experienced by the patient, in small phrases, to the point, precise.
2. Diseases: Diagnoses and diseases the patient has as mentioned in the notes.

Instructions:
1. Carefully read the provided clinical discharge summary note.
2. Extract relevant information from each of sections listed above, if present.
3. Categorize the information into the corresponding sections in the JSON format.
4. Present your analysis in this JSON format:

{
  "Symptoms": [],
  "Diseases": []
}

Important:
- Do not include the ICD-9-CM codes.
- Include all sections in the JSON, even if empty.
- Keep each symptoms as 2-3 words only.
- Keep each disease name in ICD-9 LONG_TEXT format if possible
- Include a disease or symptom only if the patient faced it, any negation of a condition shouldn't be included.
- Put relevant phrases or terms in appropriate section arrays.
- If a section is not present in the note, leave its array empty.
- Extract key information concisely, focusing on medically relevant details.
- Do NOT add any information that is not explicitly stated in the input text.

Now, analyze and categorize the following clinical admission note using the specified JSON format:

"""


test_df = pl.read_csv('clinical-outcome-prediction/saved_files/DIA_GROUPS_3_DIGITS_adm_test.csv').to_pandas()
limited_df = test_df.tail(1000)
limited_df.head()

limited_df.id.nunique()

failed_rows = []
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)


output_dir = "samples_latest"
os.makedirs(output_dir, exist_ok=True)

failed_rows = []

for i, j in tqdm(limited_df.iterrows(), total=limited_df.shape[0], desc="Processing rows"):
    notes = j.iloc[-2]

    messages = [
        {"role": "system", "content": f"{sys_prompt}"},
        {"role": "user", "content": f"### Clinical text :\n {notes} .\n"},
    ]

    try:
        all_figures = text_generator(messages, max_new_tokens=600, num_return_sequences=1, temperature=0.1)[0]['generated_text'][-1]['content']
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

        try:
            json_format = json.loads(all_figures)
            json_format['id'] = j['id']
            json_format['icd_9_desc'] = j['long_texts']
            file_path = f"{output_dir}/output_{j['id']}.json"

            with open(file_path, 'w') as file:
                json.dump(json_format, file, indent=4)

        except KeyError as e:
            failed_rows.append((i, f"KeyError: {str(e)}"))
        except json.JSONDecodeError as e:
            failed_rows.append((i, f"JSONDecodeError: {str(e)}"))
        except Exception as e:
            failed_rows.append((i, str(e)))

    except torch.cuda.OutOfMemoryError as e:
        failed_rows.append((i, "CUDA OutOfMemoryError"))
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        failed_rows.append((i, str(e)))

    gc.collect()

if failed_rows:
    print("Failed to process the following rows:")
    for row in failed_rows:
        print(f"Row index: {row[0]}, Error: {row[1]}")
