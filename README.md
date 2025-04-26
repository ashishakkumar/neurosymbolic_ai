# Neurosymbolic Approach to Improve Performance of Neural Diagnosis Systems

This repository contains the implementation for "Neurosymbolic Approach to Improve Performance of Neural Diagnosis Systems Leveraging Pubmed Based Knowledge Graph" research.

## Overview

This research introduces a neurosymbolic approach to improve the performance of neural diagnosis prediction systems. The key components include:

1. A PubMed-based Medical Knowledge Graph (PMKG) connecting symptoms with ICD-9 diseases
2. Medical Fine-tuned Language Models (MFLMs) for diagnosis prediction
3. A neurosymbolic framework combining PMKG with neural models

Our approach demonstrates significant improvements (16-67% in F1-score) across various neural models (MFLMs, LLMs, RAG) by augmenting them with structured medical knowledge.

## Repository Structure

### Knowledge Graph Construction

- **z-kg-creation/**: Contains scripts for building the PubMed-based Medical Knowledge Graph (PMKG)
  - `kg_train*.ipynb`: Notebooks for creating and training the knowledge graph using PubMed data
  
### Data Preprocessing

- **z-mimic-preprocessing/**: Scripts for processing MIMIC-III/IV datasets
  - `mimic-iv-processing.ipynb`: Preprocessing pipeline for MIMIC-IV data
  - `main2.ipynb`: Main preprocessing pipeline for MIMIC-III data
  - `ALL_3_DIGITS_DIA_CODES.txt`: List of 3-digit ICD-9 diagnosis codes used

- **z-mimic-3-files/**: Contains processed MIMIC-III datasets
  - `DIA_GROUPS_3_DIGITS_adm_train.csv`: Training data with 3-digit ICD-9 codes
  - `DIA_GROUPS_3_DIGITS_adm_test.csv`: Test data
  - `DIA_GROUPS_3_DIGITS_adm_val.csv`: Validation data

- **z-extracted-symptoms/**: Extracted symptoms from clinical notes
  - `mimic-3/`: Symptoms extracted from MIMIC-III clinical notes
  - `mimic-4/`: Symptoms extracted from MIMIC-IV clinical notes

### Model Training & Fine-tuning

- **z-finetune/**: Scripts for fine-tuning various language models
  - `biobert_finetune.py`, `biolink_finetune.py`: Fine-tuning for BioBERT and BioLinkBERT models
  - `fine_tuning.ipynb`: General fine-tuning notebook
  - `fine_tuning_llama.py`: Fine-tuning script for LLaMA models
  - `fine_tuning_gemma.ipynb`: Fine-tuning for Gemma models

### Inference & Evaluation

- **z-inference/**: Scripts for model inference
  - `infer.py`, `infer2.py`: General inference scripts
  - `mistral_inference*.py`: Inference scripts for Mistral models
  - `MIMIC_3_4_CLRAG_PMKG_SYM_EHR.ipynb`: Clinical RAG with PMKG inference

- **z-testing/**: Testing and evaluation scripts
  - `t-test.ipynb`: Statistical testing for model comparison

### Experimental Data

- **z-HSDN-rebutttal-data/**: Data files for baseline comparison with HSDN
  - Contains ICD-9 code files for comparison experiments

- **z-old-test-data/**: Legacy test data files

### Setup & Configuration

- **z-setup/**: Setup scripts for environment and infrastructure
  - `kg-setup.ipynb`: Knowledge graph setup script
  - `docker-compose.yml`: Docker configuration
  - `container-setup.ipynb`: Container setup guide

### Previous Iterations

- **z_previous_iterations/**: Previous versions of model implementation
  - Contains multiple iterations of inference notebooks

## Methodology

Our approach consists of three main components:

1. **Knowledge Graph Construction**: We build a PubMed-based Medical Knowledge Graph (PMKG) connecting symptoms from UMLS with diseases from ICD9-CM, using Neo4j.

2. **Medical Model Fine-tuning**: We fine-tune several medical language models (BioBERT, BioLinkBERT, CORe) on MIMIC-III data, using both extracted symptoms and clinical notes as input.

3. **Neurosymbolic Inference**: We combine the knowledge from PMKG with neural models' predictions to improve diagnosis accuracy:
   - Extract symptoms from clinical notes
   - Query PMKG with symptoms to find probable diseases
   - Provide symptoms, EHR notes, and PMKG-derived diseases to the model for final prediction

## Results

Our neurosymbolic approach significantly improves the performance of various models:
- Medical Fine-tuned LMs: +58-67% in F1-score
- LLMs (zero-shot): +24-27% in F1-score  
- LLMs (in-context learning): +16-20% in F1-score
- RAG framework: +19-37% in F1-score

The improvements are particularly significant for cases with multiple symptoms.

## Requirements

The project has several dependencies. Core requirements include:

```
fastapi==0.109.1
uvicorn==0.27.0
transformers==4.37.2
torch==2.2.0
pydantic==2.6.1
python-multipart==0.0.6
```

Additional dependencies needed for different components:

- **Knowledge Graph Construction**:
  - Neo4j
  - py2neo
  - pyserini
  - negspacy

- **Model Training**:
  - transformers
  - torch
  - datasets
  - accelerate
  - bitsandbytes (for quantization)

- **Inference**:
  - Flask (for API)
  - pandas
  - numpy
  - scikit-learn
  - matplotlib (for visualization)

- **MIMIC Data Processing**:
  - pandas
  - numpy
  - jupyter

## Citation

```
# Citation information will be added when published
``` 