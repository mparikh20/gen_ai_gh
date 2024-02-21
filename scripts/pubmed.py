# About this module
'''
Overall use:

## LEFT - add email or other registration requirements from NCBI
1. function that gets pmids - (caveat-limitation is only 10K for PubMed. E-direct command line utility can do >10K)
2. function that takes in pmids and returns XML with abstracts
3. function that extracts data from XML and saves it in a df, saves it.

Deep dive into how the code was developed:

'''

# imports
import html
import pandas as pd
import requests
import xml.etree.ElementTree as ET


# Specify global variables
base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
database = 'pubmed'

def get_pmids(query: str,
              save_on_server: str,
              search_format: str,
              starting_index: str,
              max_records: int):

    '''
    Description:
        Uses the esearch utility to get PMIDs matching a search query
    Args:
        query = the query string to search for. It's a good idea to first test out the string on PubMed.
        save_on_server = this is for the usehistory field within the search string shown below.
                         if 'y' : then it will save the results on the server which can be then retrieved for use in the efetch utility.
                         if '' : empty string means it will not save all the max. possible IDs on the server.
                                 here, ids will have to be retrieved directly from the resulting json generated in this function.
        search_format = type of output format. XML is default but json is possible.
        starting_index = the starting index of the first paper in the search results. eg. 0
        max_records = max. number of records to obtain.
                      If nothing is given, it will only return The default is 20 and the max. allowable IDs returned are 10,000.
    Returns:
        a json object with details about the search and the search results

    '''

    # Search query
    # the usehistory=y will save the results on a server, to be used immediately to get abstracts using the efetch utility
    search_suffix = (f'esearch.fcgi?db={database}'
                     f'&term={query}'
                     f'&usehistory={save_on_server}'
                     f'&retmode={search_format}'
                     f'&retstart={starting_index}'
                     f'&retmax={max_records}')

    # Construct url
    search_url = base_url + search_suffix

    # Make the API call
    response = requests.get(search_url)

    if response.status_code != 200:
        print(f'Status code = {response.status_code}')

    if response.encoding != 'UTF-8':
        print(f'Encoding is {response.encoding}!')


    # Get the json result and collect some metadata
    search_output = response.json()

    num_result = search_output['esearchresult']['count']
    num_ids = search_output['esearchresult']['idlist']

    print(f'The actual total number of records matching the search for {query} is {num_result}')
    print(f'The number of ids retrieved using the esearch utility is {num_ids}')
    print('Function get_pmids complete.')

    return search_output

def get_abstracts(search_output,
                  content_type: str,
                  ids_from_server: bool):
    '''
    Description:
        Takes the json output returned by the esearch utility and uses the efetch utility to return an xml string with abstracts for PMIDs related to a query.
    Args:
        search_output = json output from the get_pmids function
        content_type = a string representing the type of data or content to be collected for all the PMIDs.
                       This should be a valid entry for the efetch utility -
                       Please see db=pubmed section in https://www.ncbi.nlm.nih.gov/books/NBK25499/table/chapter4.T._valid_values_of__retmode_and/?report=objectonly
        ids_from_server = boolean
                          True means it will use specific identifier keys from the esearch json output to get all the PMIDs from the NCBI server.
                          False means it will extract the list of IDs from the json output directly.
    Returns:
        Depending on the content_type (here-abstracts), it will return data for all PMIDs in 1 XML string.
    '''

    # Option 1 - Collect PMIDs from the server and create the consequent url suffix
    if ids_from_server == True:
        query_key=search_output['esearchresult']['querykey']
        web_env = search_output['esearchresult']['webenv']

        fetchurl_suffix = f'efetch.fcgi?db={database}&query_key={query_key}&WebEnv={web_env}&rettype={content_type}&retmode=xml'

    # Option 2 - If IDs were not saved on the server while using the esearch utility, extract all IDs as a list of string and pass them to the url

    else:
        ids_of_interest = ",".join(search_output['esearchresult']['idlist'])
        fetchurl_suffix = f'efetch.fcgi?db={database}&id={ids_of_interest}&rettype={content_type}&retmode=xml'

    # Make the API call
    fetch_url = base_url + fetchurl_suffix

    fetch_response = requests.get(fetch_url)

    if fetch_response.status_code != 200:
        print(f'Status code = {fetch_response.status_code}')

    # Get the XML output with all the PMIDs and their abtracts
    pmid_data = fetch_response.text

    print('Function get_abstracts complete.')

    return pmid_data

def get_data_from_xml(pmid_data):
    '''
    Description:

    Args:
        pmid_data = xml string containing all metadata and abtracts for multiple PMIDs

    Returns
    '''

    # The XML string could have many html characters such as &#xa0 and &#x3ba and these may not be properly decoded by python
    # Clean up most of such strings using the html module

    cleaned_xml = html.unescape(pmid_data)

    # Check if html characters are cleaned up
    print(f'&#xa0 left: {cleaned_xml.count("&#xa0")}')
    print(f'&#x3ba left: {cleaned_xml.count("&#x3ba")}')
    print(f'&# left: {cleaned_xml.count("&#")}')

    # Collect information from all the articles into a dictionary
    data = {'article_title': [],
        'abstract': [],
        'pmid': [],
        'journal': [],
        'publication_type': [],
        'publication_date': [],
        'keywords':[]}

    # Use the XML module and extract information from each paper
    root = ET.fromstring(pmid_data)

    # iterate through each paper
    for child in root.iter('PubmedArticle'):

        # from each paper, get the information of interest : title, journal, publication date, abstract, keywords.
        # First create a variable to collect each piece of info.
        # In case a paper doesn't have title, then you still want an empty string to be added to the df.
        # If a variable is not specified, then it will skip that paper and create a mismatch in the datapoint numbers.
        title_val = ''
        abstract_val = ''
        pmid_val = ''
        journal_val = ''
        pubtype_val = ''
        pubdate_val = ''
        keywords_val = ''

        for title in child.iter('ArticleTitle'):
            title_val += title.text

        for abstract in child.iter('AbstractText'):
            if abstract.get('Label','x') != 'x':
                label = abstract.get('Label')
                abstract_val += f'[{label}]'
                abstract_val += abstract.text
            else:
                abstract_val += ''.join(abstract.itertext())

        for pmid in child.iter('PMID'):
            pmid_val += pmid.text

        for journal in child.iter('Title'):
            journal_val += journal.text

        for pubtype in child.iter('PublicationTypeList'):
            pubtype_val += '|'.join(pubtype.itertext())

        for pubdate in child.iter('PubDate'):
            pubdate_val += ' '.join(pubdate.itertext())

        for keyword in child.iter('KeywordList'):
            keywords_val += '|'.join(keyword.itertext())

        # Collect everything from a single paper in the dictionary
        data['article_title'].append(title_val)
        data['abstract'].append(abstract_val)
        data['pmid'].append(pmid_val)
        data['journal'].append(journal_val)
        data['publication_type'].append(pubtype_val)
        data['publication_date'].append(pubdate_val)
        data['keywords'].append(keywords_val)

    # Create a df
    papers_df = pd.DataFrame(data)

    return papers_df
