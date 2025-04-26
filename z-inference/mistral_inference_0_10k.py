from mistralai.models.jobs import TrainingParameters
from mistralai.client import MistralClient
import pandas as pd
import os
import json
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import pandas as pd

model= "open-mistral-7b" 
MISTRAL_API_KEY =  "WRhxY4qx7jsun5iYThrdS3Dk4dubsjnV"
client = MistralClient(api_key=MISTRAL_API_KEY)
train_dataset = pd.read_csv('/home/ashish/llama_inference/clinical-outcome-prediction/saved_files/DIA_GROUPS_3_DIGITS_adm_train.csv')
icd_9_df = pd.read_csv('/home/ashish/llama_inference/physionet.org/files/mimiciii/1.4/D_ICD_DIAGNOSES.csv')
all_symptoms = {}
train_dataset = train_dataset.iloc[:10000]
from tqdm import tqdm
for i,row in tqdm(train_dataset.iterrows()) : 
    sys_prompt_template =  """
You are an AI assistant analyzing clinical discharge summary notes from the MIMIC III dataset. Focus on these key sections:

>>  Symptoms: It is a subjective indication of a disease or medical condition that is experienced and reported by the patient. Unlike signs, which are objective and can be observed or measured by healthcare professionals,symptoms are personal experiences that only the individual can perceive.

>> Instructions:
1. Carefully read the provided clinical discharge summary note.
2. Categorize the information into the corresponding sections in the JSON format.
3. Present your analysis in this JSON format:

{
  "Symptoms": []
}

>> Important:
- Keep only the Symptoms as a key in JSON format.
- Keep each symptoms clear and 2-3 words only,  adhering to medical ontology standards.
- Extract key information concisely, focusing on medically relevant details.
- Do NOT add any information that is not explicitly stated in the input text.

Now, analyze and categorize the following clinical admission note using the specified JSON format:

"""
    clinical_note = f"{row['discharge_summary']}"
    sys_prompt_template += clinical_note
    prompt_template = sys_prompt_template
    
    chat_response = client.chat(
        model=model,
        max_tokens=1000,
        temperature=0.1,
        messages=[ChatMessage(role="user", content=prompt_template)]
    )
    all_symptoms[row.id] = chat_response.choices[0].message.content
    try : 
        with open(f'train_symptoms/{row.id}.json', 'w') as f:
            json.dump(json.loads(chat_response.choices[0].message.content), f)
    except : 
        continue
        
    # print(chat_response.choices[0].message.content)
    # break
    
    