'''
Overview


'''

# imports
import json
import os
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import openai_api
import pandas as pd
import tiktoken
import utils

# specify global variables
# this is the recommended model for fine-tuning
MODEL = 'gpt-3.5-turbo-0125'
TOKENS_PER_MESSAGE = 3
TOKEN_LIMIT = 16385
COST_TRAIN_TOKEN = (8/1000000)
TARGET_EPOCHS = 3
MIN_DEFAULT_EPOCHS = 1
MAX_DEFAULT_EPOCHS = 25
MIN_TARGET_EXAMPLES = 100
MAX_TARGET_EXAMPLES = 25000


def load_json(json_path):

    with open(json_path,"r") as json_file:
        data = json.load(json_file)

        return data

def get_test_set_ids(test_df,
                     id_column):
    '''
    '''
    test_df.drop_duplicates(inplace=True)

    test_ids = test_df[id_column].unique().tolist()

    return test_ids

def rows_to_dict(df):
    '''
    '''
    # Get a list of dictionaries, with each dictionary being a row
    df_dict = df.to_dict(orient='records')

    # Make an empty list to collect all dictionaries
    data_list = []

    data_dict = {}
    for data in df_dict:
        pmid = data.pop('pmid')

        if pmid not in data_dict:
            data_dict[pmid] = [data]
        else:
            data_dict[pmid].append(data)

    for key,val in data_dict.items():
        data_list.append({key:val})

    return data_list

def create_one_example(system_data:dict,
                       user_data:dict,
                       output_dict:dict,
                       pubmed_df):
    '''

    '''
    # From the output dictionary, get the PMID and get the matching abstract
    pmid = next(iter(output_dict.keys()))

    # Get the abstract string for a PMID
    abstract_text = pubmed_df[pubmed_df['pmid']==pmid]['abstract'].values[0]

    # Clean the string if required
    cleaned_abstract = openai_api.clean_string(abstract_text)

    # Add the PMID and abstract to the base user message
    text_to_append = '\n' + f'{str(pmid)}:{cleaned_abstract}'

    updated_user_data = user_data.copy()
    updated_user_data['content'] += text_to_append

    # Construct the assistant message using the output
    assistant_data = {"role":"assistant","content":json.dumps(output_dict)}

    # Construct the messages list which will go into the chat completions API as an argument
    messages = [system_data,updated_user_data,assistant_data]
    example = {"messages":messages}

    return example

def prepare_train_val_data(main_df,
                           test_ids:list,
                           system_path:str,
                           user_path:str,
                           pubmed_df,
                           training_path,
                           test_path):
    '''
    '''
    # Remove test set rows
    training_df = main_df[~main_df['pmid'].isin(test_ids)].copy()

    # Remaining ids will go into the test set
    test_df = main_df[main_df['pmid'].isin(test_ids)].copy()

    # Get a list of dictionaries from the dfs
    training_list = rows_to_dict(training_df)

    test_list = rows_to_dict(test_df)

    # Load the json file containing the system message
    system_data = load_json(system_path)

    # Load the json file containing the starting user message or instructions.
    user_data = load_json(user_path)

    # Create lists for storing all example dictionaries.
    training_examples = []
    test_examples = []
    # Loop through each output/example dictionary and create a message dictionary from each.

    for output_dict in training_list:
        # Create a single example message dictionary and store it in a list
        example = create_one_example(system_data=system_data,
                                     user_data=user_data,
                                     output_dict=output_dict,
                                     pubmed_df=pubmed_df)

        training_examples.append(example)

    for output_dict in test_list:
        example = create_one_example(system_data=system_data,
                                     user_data=user_data,
                                     output_dict=output_dict,
                                     pubmed_df=pubmed_df)

        test_examples.append(example)

    # Create jsonl files
    with open(training_path, "w", encoding="utf-8") as trainfile:
        for item in training_examples:
            json.dump(item, trainfile)
            trainfile.write('\n')

    with open(test_path, "w", encoding="utf-8") as testfile:
        for item in test_examples:
            json.dump(item, testfile)
            testfile.write('\n')

def check_data_formatting(jsonl_path):
    '''
    Copied from openAI's website - their code checks if the jsonl files are formatted properly for the model.
    '''
    # Load the dataset
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]

    # Initial dataset stats
    print("Num examples:", len(dataset))
    print("First example:")
    for message in dataset[0]["messages"]:
        print(message)

    # Format error checks
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            function_call = message.get("function_call", None)

            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")
def get_tokens_per_text(text:str,
                        model=MODEL):
    '''
    Description: Calculates tokens from a string.
    Args:
        text = string
        model = string indicating the model to use. This is already defined as a global variable above.
    Returns: Number of tokens based on the input string
    '''
    # Load the encoding model specific to the model, here: gpt-4-turbo-preview (this will usually point to their latest model)
    encoding = tiktoken.encoding_for_model(model)

    num_tokens = len(encoding.encode(text))

    return num_tokens

def get_tokens_per_message(message_dict:dict,
                           add_tokens_per_message:bool,
                           model=MODEL):
    '''
    Description: Calculates tokens for all values within a message (contained within a dicitonary.)
    Args:
        message_dict = dictionary with the typical 'role' and 'system' keys or according to the openai format.
        add_tokens_per_message = boolean indicating whether to add priming tokens or not.
                                 This value is already set as a global variable above.
    Returns: number of tokens for an entire message dictionary
    '''

    # Calculate tokens
    num_tokens = 0
    for value in message_dict.values():
        num_tokens += get_tokens_per_text(text=value,model=model)

    # Add priming tokens per message if required
    if add_tokens_per_message:

        num_tokens += TOKENS_PER_MESSAGE

    return num_tokens

def estimate_finetuning_cost(jsonl_path:str):
    '''
    '''
    # Load the jsonl file into a list of messages dictionaries, each one pointing to 1 example
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        examples = [json.loads(line) for line in f]

    token_limit_exceeded = 0
    token_values = []

    for example in examples:
        # messages is a list of 3 dictionaries - system,user, and assistant
        messages = example.get('messages')
        tokens_per_example = 0
        # For each message dictionary, calculate tokens
        for message in messages:
            num_tokens = get_tokens_per_message(message_dict=message,
                                                add_tokens_per_message=True,
                                                model=MODEL)
            tokens_per_example += num_tokens

        if tokens_per_example >= TOKEN_LIMIT:
            token_limit_exceeded += 1

        token_values.append(tokens_per_example)

    total_tokens = sum(token_values)
    print(f'Max tokens in 1 example: {max(token_values)}')
    print(f'Number of examples where token limit exceeded {TOKEN_LIMIT}: {token_limit_exceeded}')
    print(f'Total number of tokens: {total_tokens}')

    n_examples = len(examples)

    print(f'There are {n_examples} examples in the finetuning dataset.')

    n_epochs = TARGET_EPOCHS
    if n_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_examples)

    elif n_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_examples)

    print(f'Based on default epoch and training example numbers, epochs should be {n_epochs}.')

    tokens_charged = total_tokens * n_epochs

    total_cost = round(COST_TRAIN_TOKEN * tokens_charged,2)

    print(f'Based on the finetuning cost of ${COST_TRAIN_TOKEN} per token, estimated cost is ${total_cost} for a dataset with {n_examples} examples and {total_tokens} tokens.')

def upload_file(jsonl_path:str,
                openai_key_path:str):
    '''
    '''
    # Setup client
    client = openai_api.setup_client(openai_key_path)

    # Load the file and collect the response
    upload_response = client.files.create(file=open(jsonl_path, "rb"),purpose="fine-tune")

    return upload_response

def finetune_model(openai_key_path:str,
                   training_file_id:str,
                   hyperparameters=None,
                   model_suffix=None,
                   validation_file_id=None):
    '''
    '''
    # Setup client
    client = openai_api.setup_client(openai_key_path)

    finetuning_object = client.fine_tuning.jobs.create(model=MODEL,
                                                       training_file=training_file_id,
                                                       hyperparameters=hyperparameters,
                                                       suffix=model_suffix,
                                                       validation_file=validation_file_id)

    return finetuning_object

def get_df_from_completions(completions_dict:dict):
    '''
    '''
    # Collect all the df rows as dicts in this main list
    main_data = []

    # Each completion ID points to a dictionary of dictionaries
    for key,value in completions_dict.items():
        main_row = {}
        main_row['completion_id'] = key

        # Lopp through each dictionary and collect relevant info
        for key,result in value.items():
            if 'usage' in key:
                main_row = main_row | result
            if 'model' in key:
                main_row[key] = result
            if 'data' in key:
                for data_key,data_val in result.items():
                    if data_key.startswith('PMID'):
                        pmid_val = data_key[4:]
                    else:
                        pmid_val = data_key

                    main_row['pmid'] = pmid_val
                    if len(data_val) == 0 or data_val is None:
                        data_val = [{}]
                    for drug_dict in data_val:
                        add_row = main_row | drug_dict
                        main_data.append(add_row)

    main_df = pd.DataFrame(main_data)

    # Replace empty lists to None
    main_df = main_df.applymap(lambda x: None if isinstance(x, list) and len(x) == 0 else x)

    # Replace lists containing null string to None
    main_df = main_df.applymap(lambda x: None if isinstance(x, list) and x[0]=='null' else x)

    main_df.replace({np.nan:None}, inplace=True)

    return main_df

def get_finetuning_results(openai_key_path:str,
                           finetuning_job_id:str,
                           ft_results_path:str):

    # Set up API client
    client = openai_api.setup_client(openai_key_path)

    # Retrieve finetuning job details
    results_object = client.fine_tuning.jobs.retrieve(finetuning_job_id)

    # Extract relevant details
    model_id = results_object.fine_tuned_model
    base_model = results_object.model
    hyperparameters = str(results_object.hyperparameters)
    seed = results_object.seed
    # This will be an array of file ID(s)
    ft_results_files = results_object.result_files

    # Get file ID
    ft_results_file_id = ft_results_files[0]

    # Get file contents - everything will be contained in 1 string
    content = client.files.retrieve_content(ft_results_file_id)

    # Split the string at newlines giving a list of strings
    content_list = content.split("\n")

    # Remove empty strings
    for string in content_list:
        if len(string)==0:
            content_list.remove(string)

    # Each string represents a row that needs to be further split into a list.
    # Generate a list of lists, each list being 1 row
    content_list = [string.split(",") for string in content_list]

    # Add information to each line in the list
    for count,row in enumerate(content_list):
        if 'step' in row:
            row = row + ['model_id','base_model','seed','hyperparameters']
        else:
            row = row + [model_id,base_model,seed,hyperparameters]

        content_list[count] = row

    # Collect the column names
    cols = content_list.pop(0)

    df = pd.DataFrame(content_list,columns=cols)

    df.to_csv(ft_results_path,index=False,encoding='utf-8')
