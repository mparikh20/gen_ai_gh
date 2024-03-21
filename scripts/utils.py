
'''
Overview
This module has functions that can be used with Google APIs or other data sources.

Refs:
This article has instructions on setting up service accounts in the GCP console:
https://medium.com/@jb.ranchana/write-and-append-dataframes-to-google-sheets-in-python-f62479460cf0
'''

# imports
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd

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
