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
import pandas as pd
import utils

# specify global variables
# MODEL = 'gpt-4-turbo-preview'

def load_json(json_path):

    with open(json_path,"r") as json_file:
        data = json.load(json_file)

        return data

def clean_string(text):
    '''
    Description: Takes a string and performs a cleaning step to remove any trailing spaces and double spaces.
    Args:
        text = string
    Returns: cleaned up string
    '''

    if isinstance(text,str):

        # remove leading or trailing spaces and replace double spaces with a single space.
        text = text.strip().replace("  "," ")

    return text

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
    cleaned_abstract = clean_string(abstract_text)

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

