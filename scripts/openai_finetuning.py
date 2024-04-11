'''
Overview


'''

# imports
import json
import os
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
                           training_fraction:float,
                           system_path:str,
                           user_path:str,
                           pubmed_df,
                           training_path,
                           validation_path):
    '''
    '''
    # Remove test set rows
    main_df2 = main_df[~main_df['pmid'].isin(test_ids)].copy()

    # Randomly select a fraction of pmids that should be for training
    train_ids = main_df2[['pmid']].drop_duplicates().sample(frac=training_fraction,random_state=1,replace=False)['pmid'].unique()

    # Get training df
    training_df = main_df2[main_df2['pmid'].isin(train_ids)].copy()

    # Remaining ids will go into the validation set
    validation_df = main_df2[~main_df2['pmid'].isin(train_ids)].copy()

    # Get a list of dictionaries from the dfs
    training_list = rows_to_dict(training_df)

    validation_list = rows_to_dict(validation_df)

    # Load the json file containing the system message
    system_data = load_json(system_path)

    # Load the json file containing the starting user message or instructions.
    user_data = load_json(user_path)

    training_examples = []

    validation_examples = []
    # Loop through each output/example dictionary and create a message dictionary from each.
    for output_dict in training_list:
        # Create a single example message dictionary and store it in a list
        example = create_one_example(system_data=system_data,
                                     user_data=user_data,
                                     output_dict=output_dict,
                                     pubmed_df=pubmed_df)

        training_examples.append(example)

    for output_dict in validation_list:
        example = create_one_example(system_data=system_data,
                                     user_data=user_data,
                                     output_dict=output_dict,
                                     pubmed_df=pubmed_df)

        validation_examples.append(example)

    # Create jsonl files
    with open(training_path, "w", encoding="utf-8") as trainfile:
        for item in training_examples:
            json.dump(item, trainfile)
            trainfile.write('\n')

    with open(validation_path, "w", encoding="utf-8") as valfile:
        for item in validation_examples:
            json.dump(item, valfile)
            valfile.write('\n')

