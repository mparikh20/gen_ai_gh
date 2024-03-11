# imports
from dotenv import load_dotenv
import json
from openai import OpenAI
import os
import pandas as pd
from pathlib import Path
import tiktoken

# specify global variables (MAKE UPPER)
MODEL = 'gpt-4-turbo-preview'
TOTAL_TOKENS = 120000
TOKENS_PER_MESSAGE = 3
REPLY_TOKENS = 3
COST_PER_INPUT_TOKEN = 0.00001
COST_PER_OUTPUT_TOKEN = 0.00003

def select_pubmed_data(data_df,
                       cols:list,
                       num_rows=None,
                       random_state=None)->pd.DataFrame:
    '''
    Description: Takes a df and selects only the data of interest form which inputs will be generated for the llm api.
    Args:
        data_df = df containing data of interest
        cols = a list of column names that contain relevant information to be used in crafting prompts for the LLM.
        num_rows = integer. Can be none if all the rows of the df should be used.
            If rows should be randomly sampled, enter the number of rows to sample from the df.
        random_state = int, a seed for sampling the rows.
                       Use the same number if the same rows need to be selected in repeat runs.
    Returns: cleaned up string
    '''
    if num_rows is None:
        # Select columns of interest and rows with information
        data_df1 = data_df[~data_df[cols].isna()][cols]

    else:
        data_df1 = data_df[~data_df[cols].isna()][cols].sample(n=num_rows,random_state=random_state)

    print('Function select_pubmed_data complete.')

    return data_df1

def clean_string(text):
    '''
    Description: Takes a string and performs a cleaning step to remove any trailing spaces and double spaces.
    Args:
        text = string
    Returns: cleaned up string
    '''

    if type(text) == str:

        # remove leading or trailing spaces and replace double spaces with a single space.
        text = text.strip().replace("  "," ")

    return text

def get_tokens_per_text(text:str,model=MODEL):
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
    for key,value in message_dict.items():
        num_tokens += get_tokens_per_text(text=value,model=model)

    # Add priming tokens per message if required
    if add_tokens_per_message:

        num_tokens += TOKENS_PER_MESSAGE

    return num_tokens

def setup_client(openai_key_path):
    '''
    Description: Gets the API key and sets up the openai API client.
    Args:
        openai_key_path = str, path to the key
    Returns:
        client for calling the API
    '''
    # Load the API key
    load_dotenv(dotenv_path=openai_key_path)

    # Access the key
    openai_api_key = os.environ['OPENAI_API_KEY']

    client = OpenAI(api_key=openai_api_key)

    print('Function setup_client complete.')
    return client

def call_api(client,
             messages,
             seed:int,
             temperature:float):
    '''
    Description: Makes 1 API call, gets the completion json object, and extracts only specific details from it into a dictionary.
    Args:
        client = client setup to call the API (output of setup_client function)
        messages = a list of dictionaries, each representing 1 message according to the openai format
        seed = a seed to make sure the same output type is generated upon re-running the same prompt (this is not always guaranteed)
        temperature = a float in the range of 0-2 indicating how creative the responses should be.
                      Use lower temperatures for consistent & conservative response and higher values for creative responses.
    Returns: (1) API call id and (2) a dictionary with usage, task completion output, and model used
    '''

    temperature = round(float(temperature),1)

    # Check the temperature parameter is within allowable range:
    if 0.0 <= round(temperature,1) <= 2.0:
        response = client.chat.completions.create(model=MODEL,
                                                  response_format={ "type": "json_object" },
                                                  messages=messages,
                                                  seed=seed,
                                                  temperature=temperature
                                              )
        # The response JSON object can be loaded as a Python object
        result = json.loads(response.model_dump_json())

        # Make a dictionary to store relevant details
        output = {}

        # Extract relevant details
        output['usage_dict'] = result['usage']

        output['data_dict'] = json.loads(result['choices'][0]['message']['content'])

        output['model_str'] = result['model']

        output_id = result['id']

        print('Function call_api complete.')
        return output_id, output

    else:
        print(f'Input temperature {temperature} is out of range 0-2.')

def process_pubmed_data(system_path: str,
                        user_path: str,
                        data_df1: pd.DataFrame,
                        openai_key_path: str,
                        seed:int,
                        temperature:float)->dict:
    '''
    Description: Takes a df containing pubmed data and makes an API call for completing specified tasks for each PMID.
    Args:
        system_data = str, path to the json file containing the system message
        user_path = str, path to the json file containing the system message
        data_df1 = df containing pmid and abstracts
        openai_key_path = path to the API key
        seed = a seed to make sure the same output type is generated upon re-running the same prompt (this is not always guaranteed)
        temperature = a float in the range of 0-2 indicating how creative the responses should be.
                      Use lower temperatures for consistent & conservative response and higher values for creative responses.
    Returns:

    '''

    # Load the json file containing the system message
    with open(system_path, "r") as json_file:
        system_data = json.load(json_file)

    # Load the json file containing the starting user message or instructions.
    with open(user_path, "r") as json_file:
        user_data = json.load(json_file)

    # Get number of tokens for the base message and get an updated starting value for number of tokens

    system_tokens = get_tokens_per_message(message_dict=system_data,
                                           add_tokens_per_message=True,
                                           model=MODEL)

    user_tokens = get_tokens_per_message(message_dict=user_data,
                                         add_tokens_per_message=True,
                                         model=MODEL)

    # Add all tokens coming from the base message as well as some tokens alloted from the completion.
    calculated_tokens = system_tokens + user_tokens

    # Confirm the following columns are present in the df
    if 'pmid' in data_df1.columns and 'abstract' in data_df1.columns:
        print('\tNecessary columns present in the input df.')

    else:
        print('\tOne or both of the pmid and abstract columns are not present in the input df. Exiting the function.')
        return

    # Make a list of PMIDs to keep track of which ones get added to the prompt and which ones should go to the next API call
    pmids = data_df1['pmid'].values

    # Make a dictionary to collect completions for each API call.
    all_completions = {}

    # Track cost
    cost = 0.00

    # Setup client
    client = setup_client(openai_key_path)

    # Get corresponding abstract from the df.
    for pmid in pmids:

        # Get the abstract string for a PMID
        abstract_text = data_df1[data_df1['pmid']==pmid]['abstract'].values[0]

        # Clean the string if required
        cleaned_abstract = clean_string(abstract_text)

        # Add the ID in front of the text
        text_to_append = '\n' + f'PMID{str(pmid)}:{cleaned_abstract}'

        # Get token count from this abstract
        abstract_tokens = get_tokens_per_text(text=text_to_append,model=MODEL)

        # Check if adding the text will keep it within the token limit
        if (calculated_tokens + abstract_tokens + REPLY_TOKENS) <= TOTAL_TOKENS:
            # If yes, append the text to the user prompt
            updated_user_data = user_data.copy()

            updated_user_data['content'] += text_to_append

            # Construct the messages list which will go into the chat completions API as an argument
            messages = [system_data,updated_user_data]

            # Call the API
            output_id, output = call_api(client=client,
                                         messages=messages,
                                         seed=seed,
                                         temperature=temperature)

            # Add the dictionary output with it's API call identifier
            all_completions[output_id] = output

            # Get actual cost
            cost += (output['usage_dict']['completion_tokens'] * COST_PER_OUTPUT_TOKEN) + (output['usage_dict']['prompt_tokens'] * COST_PER_INPUT_TOKEN)

        else:
            print(f'Skipping PMID {pmid} due to token limitations.')

    print(f'Function process_pubmed_data complete. The cost of the run is {round(cost,2)}.')
    return all_completions
