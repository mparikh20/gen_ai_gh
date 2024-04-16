
'''
Overview
This module has functions that can be used with Google APIs or other data sources.

Refs:
This article has instructions on setting up service accounts in the GCP console:
https://medium.com/@jb.ranchana/write-and-append-dataframes-to-google-sheets-in-python-f62479460cf0
'''

# imports
import ast
import gspread
from google.oauth2.service_account import Credentials
import numpy as np
import pandas as pd
import string

def write_data_to_sheets(df,
                         credentials_path:str,
                         sheets_filename:str,
                         sheets_sheetname:str,
                         columns_to_write:list):
    '''
    Description: Takes data from a df and writes it to Google Sheets
    Args:
        df = df of interest
        credentials_path = path to the Google service account credentials
        sheets_filename = name of the Google Sheets file
        sheets_sheetname = name of the worksheet within the file
        columns_to_write = list of columns to write to the Sheets
    Returns: None, data written to Sheets
    '''
    # Set up credentials
    scope = ['https://www.googleapis.com/auth/spreadsheets','https://www.googleapis.com/auth/drive']

    credentials = Credentials.from_service_account_file(credentials_path, scopes=scope)
    client = gspread.authorize(credentials)

    # Open the sheet
    spreadsheet = client.open(sheets_filename)
    worksheet = spreadsheet.worksheet(sheets_sheetname)

    # Specify which columns and values to write
    values_to_write = df[columns_to_write].values.tolist()

    # Specify which cells to write to
    start_cell = 'A1'
    end_cell = chr(ord('A') + len(columns_to_write) - 1) + str(len(values_to_write) + 1)
    range_to_write = f'{start_cell}:{end_cell}'

    worksheet.update(range_to_write, [columns_to_write] + values_to_write)

def read_data_from_sheets(credentials_path:str,
                          sheets_filename:str,
                          sheets_sheetname:str,
                          start_cell:str,
                          end_cell:str):
    '''
    Description: Takes data from a Google Sheet and writes it into a df. If any entire column is empty, it will first be filled with 'NA'
    Args:
        credentials_path = path to the Google service account credentials
        sheets_filename = name of the Google Sheets file
        sheets_sheetname = name of the worksheet within the file
        start_cell = starting cell in the Sheet
        end_cell = ending cell in the Sheet
    Returns: Df
    '''
    # Set up credentials
    scope = ['https://www.googleapis.com/auth/spreadsheets','https://www.googleapis.com/auth/drive']

    credentials = Credentials.from_service_account_file(credentials_path, scopes=scope)
    client = gspread.authorize(credentials)

    # Open the sheet
    spreadsheet = client.open(sheets_filename)
    worksheet = spreadsheet.worksheet(sheets_sheetname)

    # Specify which columns and values to write
    start_cell = start_cell
    end_cell = end_cell
    range_to_write = f'{start_cell}:{end_cell}'

    # Before writing to a df, check if any of the columns is completely empty
    # This will be a sring 'ABCD...'
    chars = string.ascii_uppercase

    # Iterate through each column based on the start cell
    start_column = start_cell[0]
    end_column = end_cell[0]

    start_index = chars.find(start_column)
    end_index = chars.find(end_column)

    # Total rows to read:
    starting_row = int(start_cell[1:])
    ending_row = int(end_cell[1:])

    for col in chars[start_index:end_index+1]:

        col_data = worksheet.get(f'{col}{str(starting_row)}:{col}{str(ending_row)}')

        if len(col_data) == 1:
            print(f'Column {col} has empty values and will be filled with empty strings in the Sheet.')
            # if only 1 list that will be just the column name

            # add lists with NA to fill all empty cells in the column
            # Tried this with empty string but it doesn't work
            values = [['NA'] for num in range(ending_row - starting_row)]
            cells_to_fill = f'{col}{str(starting_row+1)}:{col}{ending_row}'

            worksheet.update(values,cells_to_fill)

    # After filling empty column with empty strings, write data into a df
    data = worksheet.get(range_to_write)

    # Load the data into a df
    df = pd.DataFrame(data[1:],columns=data[0])

    return df

def convert_string_to_structure(value):
    '''
    Description: Takes a string and processes it into a Python data structure that could be contained within a string
                 eg. "['a','b','c']" string will be converted to ['a','b','c'] list
                 This is useful to pass as a 'converters' while reading a CSV (if located in the same directory) or apply after the df has been loaded
    Args:
        value = string
    Returns: relevant Python data structure
    '''
    try:
        return ast.literal_eval(value)
    except (ValueError,SyntaxError):
        return value

def clean_missing_data(df):
    '''
    Description: Replaces the following types of missing data to None values.
    Args:
        df
    Returns: Cleaned up df
    '''

    df.replace({np.nan:None,
                '':None,
                'None':None}, inplace=True)
    return df
