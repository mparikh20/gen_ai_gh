# Automated drug target relationship extraction from biomedical literature using Generative AI (2024)


## Keywords / Tags
RAG, LLM, named entity recognition, generative AI, information retrieval, finetuning, GPT, OpenAI, Python, API, PubMed, NCBI eutils, prompt engineering

## Overview

### Motivation
Biomedical literature from PubMed is a rich resource for therapeutic discoveries and can be leveraged to generate high quality drug-centric datasets. Automated extraction of entities from literature such as drugs, their targets, relationship or interaction type between the drug and targets, indications tested, and updated knowledge about therapeutics can be valuable for many use cases such as:

* real world evidence analyses
* computational analyses for target discovery and validation
* assessment of competitive landscape of drug development assets
* ground truth or gold standard data generation supporting evaluation of therapeutic discoveries coming from AI models

Although there are certain publicly available datasets containing drug-centric information, these may suffer from outdated content, incomplete or ambiguous data, and misalignment of the data types or information type with the company's technical or commercial goals.

With LLMs, it is possible to design and implement in-house pipelines using Retrieval Augmented Generation (RAG) approach enabling automated extraction of information and continued access to an updated knowledgebase.

### Approach

Published biomedical articles involving drug(s) within specific cancers of interest were obtained from PubMed. A total of 87 articles (30 breast cancer, 30 glioblastoma, and 27 lung cancer) were obtained using NCBI's E-utilities API. Feasibility was confirmed by using GPT-4 to extract drug-centric information from only abstract text from 30 articles. Second step involved fine-tuning GPT-3.5 on 57 training examples, and using the fine-tuned model for inference on the 30 abstracts (test set). Performance of each step was measured by comparing the model outputs with a manually curated ground truth dataset.

### Step 1 - Test GPT-4 for knowledge and relationship extraction and show proof of concept with 30 PubMed abstracts

To confirm feasibility of using GPT for extraction of drug-centric information, GPT-4 was tested on 30 PubMed abstracts randomly selected from the total 87 in the dataset. Base prompt language was selected from extensive experimentation involving series of testing zero-shot, one-shot, and few-shot prompt formats on hypothetical data or few abstracts that weren't a part of the dataset of 87 articles.

GPT-4 (turbo-preview; gpt-4-0125-preview) was used to extract specific entities such as drug name, target name, their relationship, clinicaltrials id, diseases tested, and sample groups tested from 30 PubMed abstracts.

For evaluation, a manually curated dataset was prepared for all the 30 abstracts. Of all the extracted information types, the following 3 were the most important and complex to identify: drug name, target, and drug-target interaction types. Hence, evaluation metrics were obtained only for these 3 data types by comparing the results from GPT-4 with a manually curated, ground truth dataset.

#### Creation of manually curated ground truth data
GPT-4 was used to create a starting dataset where prompts were given for it to extract drug name, its targets, and the interactions between drug-target pairs from each of the 30 PubMed abstracts. This initial dataset was then reviewed and annotated manually to correct any errors from GPT-4. This assistive approach was taken to shorten the time to create manually curated content from scratch.

#### Evaluation
Overall, the following strategy was used for evaluation of the GPT-4 extracted entities. GPT-4 extracted drug names, their targets, and the drug-target interactions with the following precision, recall as shown in the schematic. Even with exhaustive instructions included in the prompt, GPT-4 doesn't give good performance at all levels of evaluation (shown in the schematic below.). Since biomedical context can be pretty complex, further prompt engineering is not expected to significantly improve performance, necessitating application of a model fine-tuned on this specific dataset.

```
LEVEL 1 - Drug name
  |       For each abstract/PMID, get drug names correctly identified by GPT-4.
  |       Precision = 91%, Recall = 57%
  |
  |_ LEVEL 2 - Target
       |       For drugs that were extracted correctly, did GPT-4 identified the associated targets correctly?
       |       Precision = 75%, Recall = 60%
       |
       |_ LEVEL 3 - Drug-Target interaction
            For drug-target pairs that were extracted correctly, did GPT-4 capture their interaction type properly?
            Precision = 89%, Recall = 89%
```

### Step 2 - Fine tune GPT-3.5 on 57 PubMed abstracts and determine if finetuning shows improved performance compared to GPT-4 used without any fine-tuning.
In Step 1, GPT-4 was tested on 30 PubMed extracts. In step 2, these 30 abstracts constituted the test set. Out of the total 87 abstracts, the remaining 57 abstracts were manually curated as described above and constituted the training data. These 57 abstracts were used to fine tune GPT-3.5 (specifically, gpt-3.5-turbo-0125). Similar to step 1, performance of extraction was evaluated by focusing only on: drug name, associated targets, and the drug-target relationship from each abstract text (Refer to Creation of manually curated ground truth data and Evaluation in Step 1).

#### Evaluation
Overall, fine-tuned GPT-3.5-turbo showed a better performance in extracting drugs, and their targets compared to just prompt engineering with GPT-4. However, GPT-4 performed better at extracting the drug-target relationship. The schematic below shows the precision and recall from the 2 approaches.

```
LEVEL 1 - Drug name
  |       For each abstract/PMID, get drug names correctly identified by the model.
  |       Finetuned GPT3.5: Precision = 100%, Recall = 84%
  |       GPT4: Precision = 91%, Recall = 57%
  |
  |_ LEVEL 2 - Target
       |       For drugs that were extracted correctly, did the model identified the associated targets correctly?
       |       Finetuned GPT3.5: Precision = 100%, Recall = 71%
       |       GPT4: Precision = 69%, Recall = 51%
       |
       |_ LEVEL 3 - Drug-Target interaction
        For drug-target pairs that were extracted correctly, did the model capture their interaction type properly?
            Finetuned GPT3.5: Precision = 86%, Recall = 86%
            GPT4: Precision = 89%, Recall = 89%
```

## Conclusion
* This project shows that RAG can be used to connect LLMs to the latest published literature and this approach can be leveraged to automatically extract drug-centric entities and their relationships solely from abstracts of biomedical articles.
* Although this project was tested on a small dataset (57 examples for fine-tuning, 30 examples in the test set) due to the limitations of having manually curated ground truth data, the performance shows possibility of improvements with a higher dataset size (100+ examples for finetuning).
* Many external drug-centric databases offered by data companies or publicly available datasets are not updated very well and suffer from low precision of drug-target relationships. In contrast, implementing a RAG pipeline from specific data sources such as PubMed has a major advantage of generating an updated knowledgebase straight from published biomedical literature. Additionally, the data and information can be extracted and structured specifically to a company's technical and commercial goals.

## Repo structure and contents

- **/gen_ai_gh/** - Main directory of the project
  - **/inputs/**
    - `system_prompt_v1.json` - Prompt message for the role 'system' (Step 1, 2)
    - `test_finetuning_v1.jsonl` - Validation file used to calculate loss during finetuning (Step 2)
    - `training_finetuning_v1.jsonl` - Training examples used for finetuning GPT-3.5-turbo (Step 2)
    - `user_prompt_v1.json` - Prompt message for the role 'user' used for GPT-4 inference (Step 1)
    - `user_prompt_v2.json` - Shorter prompt message; not used in any project.
    - `user_prompt_v3.json` - Prompt message for the role 'user' used for finetuning GPT-3.5 and inference on test set (Step 1)

  - **/notebooks/** - Contains jupyter notebooks
    - `llms_playground.ipynb` - Testing and building OpenAI API workflow, prompt engineering
    - `ner_pubmed_llm.ipynb` - Shows Step 1 workflow
    - `pubmed_finetuning.ipynb` - Shows Step 2 workflow

  - **/outputs/** - Contains source files
    - `completions_run_1.json` - Completions for 30 pubmed abstracts processed for entity extraction via OpenAI API-GPT-4 (Step 1)
    - `gpt3_5_completions_run_3.json` - Completions for pubmed abstracts from the test set (n=30), inference by a finetuned GPT-3.5 (Step 2)
    - `gpt4_completions_run_2.json` - Completions from GPT-4 for 87 pubmed abstracts, for creating starting, manually curated data (Step 1,2)
    - `pubmed_data_run_1.csv` - CSV containing PubMed abstracts and other metadata for 87 articles (Step 1,2)
    - `pubmed_ft1_model_loss_metrics.csv` - Finetuning results (loss metrics) saved into a CSV along with hyperparameter info (Step 2)
    - `pubmed_ner_gpt3_5_run_3.csv` - Results from gpt3_5_completions_run_3.json from the finetuned model extracted into a CSV (Step 2)
    - `pubmed_ner_gpt4_run_2.csv` - Results from gpt4_completions_run_2.json extracted into a CSV (Step 1,2)
    - `pubmed_ner_run_1.csv` - Results from completions_run_1.json extracted into a CSV. (Step 1)

  - **/scripts/** - Contains python modules
    - `ner_evaluation.py` - Code to evaluate model extracted entities against manually curated ground truth data.
    - `openai_api.py` - Code for running a pipeline using the GPT API
    - `openai_finetuning.py` - Code for all steps within the finetuning workflow
    - `pubmed.py` - Code for querying PubMed, collecting metadata, organizing abstracts and other information into a df.
    - `utils.py` - Code for interacting with the Google Sheets API and some data cleaning functions.

## Accessory content
- **/exploratory-analysis/** - Repo for EDA / testing
    - **/working_with_apis/**
        - `testing_apis.ipynb` - Shows details about the PubMed API and development of the workflow shown in pubmed.py
