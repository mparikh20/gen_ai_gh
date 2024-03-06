# About this module
'''
Overall use:

1. function that gets pmids - (caveat-limitation is only 10K for PubMed. E-direct command line utility can do >10K)
2. function that takes in pmids and returns XML with abstracts
3. function that extracts data from XML and saves it in a df, saves it.

Deep dive into how the code was developed:

Note about API key:
Currently the code is written without pulling any API key.
Without a key, 3 requests per second are allowed.
With an API key, 10 requests per second are allowed by default.

'''

# imports
from datetime import datetime
import html
import pandas as pd
import requests
import xml.etree.ElementTree as ET


# Specify global variables
base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
database = 'pubmed'
today_date = datetime.today().date()

def get_pmids(query: str,
              save_on_server: str,
              search_format: str,
              search_starting_index: str,
              search_max_records: int,
              sorting_criteria:str):

    '''
    Description:
        Uses the esearch utility to get PMIDs matching a search query
    Args:
        query = the query string to search for. It's a good idea to first test out the string on PubMed.
                Please refer to https://pubmed.ncbi.nlm.nih.gov/help/#searching-by-date on how to query.
                As a tip, use the Advanced Search tool on the website. Pasting the query verbatim as used on the website will also work.
        save_on_server = this is for the usehistory field within the search string shown below.
                         if 'y' : then it will save the results on the server which can be then retrieved for use in the efetch utility.
                         if '' : empty string means it will not save all the max. possible IDs on the server.
                                 here, ids will have to be retrieved directly from the resulting json generated in this function.
        search_format = type of output format. XML is default but json is possible.
        starting_index = the starting index of the first paper in the search results. eg. 0
        max_records = max. number of records to obtain.
                      If nothing is given, it will only return 20. The default is 20 and the max. allowable IDs returned are 10,000.
        sorting_critera = how should results be sorted? Valid values are:
                          pub_date – descending sort by publication date
                          Author – ascending sort by first author
                          JournalName – ascending sort by journal name
                          relevance – default sort order, (“Best Match”) on web PubMed
    Returns:
        a json object with details about the search and the search results

    '''

    # Search query
    # the usehistory=y will save the results on a server, to be used immediately to get abstracts using the efetch utility
    search_suffix = (f'esearch.fcgi?db={database}'
                     f'&term={query}'
                     f'&usehistory={save_on_server}'
                     f'&retmode={search_format}'
                     f'&retstart={search_starting_index}'
                     f'&retmax={search_max_records}'
                     f'&sort={sorting_criteria}')

    # Construct url
    search_url = base_url + search_suffix

    # Make the API call
    response = requests.get(search_url)

    if response.status_code != 200:
        print(f'\tStatus code = {response.status_code}')

    if response.encoding != 'UTF-8':
        print(f'\tEncoding is {response.encoding}!')


    # Get the json result and collect some metadata
    search_output = response.json()

    num_result = search_output['esearchresult']['count']
    num_ids = len(search_output['esearchresult']['idlist'])

    print(f'\tThe actual total number of records matching the search for is {num_result}')
    print(f'\tThe number of ids present in the esearch json is {num_ids}')
    print('\tFunction get_pmids complete.')

    return search_output

def query_and_ids(search_output):

    '''
    Description: Takes the json output and builds a dictionary showing some search metadata.
    Args:
        search_output = json output from the get_pmids function
    Returns:
        A dictionary with the following info: the actual query string used for searching, total count matching the query, actual number of results retrieved,
        and all the PMIDs.
    '''

    # Add a string of all total matching PMIDs.
    id_str = ",".join(search_output['esearchresult']['idlist'])

    # Capture metadata about the search results
    # Add updated date to indicate when these were obtained

    metadata_dict = {'query_string': search_output['esearchresult']['querytranslation'],
                     'num_total_matches': int(search_output['esearchresult']['count']),
                     'all_matching_pmids':id_str,
                     'acquisition_date': today_date}

    print('\tMetadata obtained and saved in a dictionary.')

    return metadata_dict

def get_abstracts(search_output,
                  content_type: str,
                  ids_from_server: bool,
                  fetch_starting_index: str,
                  fetch_max_records: int):
    '''
    Description:
        Takes the json output returned by the esearch utility and uses the efetch utility to return an xml string with abstracts for PMIDs related to a query.
    Args:
        search_output = json output from the get_pmids function
        content_type = a string representing the type of data or content to be collected for all the PMIDs.
                       This should be a valid entry for the efetch utility - eg. 'abstract'
                       Please see db=pubmed section in https://www.ncbi.nlm.nih.gov/books/NBK25499/table/chapter4.T._valid_values_of__retmode_and/?report=objectonly
        ids_from_server = boolean
                          True means it will use specific identifier keys from the esearch json output to get all the PMIDs from the NCBI server.
                          False means it will extract the list of IDs from the json output directly.
        starting_index = the starting index of the first paper in the search results. eg. 0
                         This argument necessary only when ids_from_server is True
        max_records = max. number of records to obtain.
                      This argument necessary only when ids_from_server is True.
    Returns:
        A list of PMIDs relevant to the search query and only those for which content should be extracted (based on fetch_max_records).
        Depending on the content_type (here-abstracts), it will return data for all PMIDs in 1 XML string.
    '''

    # Option 1 - Collect PMIDs from the server and create the consequent url suffix
    if ids_from_server:
        query_key=search_output['esearchresult']['querykey']
        web_env = search_output['esearchresult']['webenv']

        fetchurl_suffix = f'efetch.fcgi?db={database}&query_key={query_key}&WebEnv={web_env}&rettype={content_type}&retstart={fetch_starting_index}&retmax={fetch_max_records}&retmode=xml'

        # Get list of all ids for which data should be collected:
        ids_suffix = f'efetch.fcgi?db={database}&query_key={query_key}&WebEnv={web_env}&rettype=uilist&retmode=text&retstart={fetch_starting_index}&retmax={fetch_max_records}'

        # API call for getting just the PMIDs
        ids_url = base_url + ids_suffix
        ids_response = requests.get(ids_url)

        # This will be a string of ids.
        ids_str = ids_response.text
        # Convert the string to a list of ids
        ids_of_interest = ids_str.strip("\n").split("\n")
        print(f'\tThe number of matching PMIDs based on the server: {len(ids_of_interest)}')

    # Option 2 - If IDs were not saved on the server while using the esearch utility, extract all IDs as a list of string and pass them to the url

    else:
        # Get a list of ids
        ids_of_interest = search_output['esearchresult']['idlist']

        # Convert it to a string of ids to pass it to the url
        ids_str = ",".join(ids_of_interest)
        fetchurl_suffix = f'efetch.fcgi?db={database}&id={ids_str}&rettype={content_type}&retmode=xml'

    # Make the API call
    fetch_url = base_url + fetchurl_suffix

    fetch_response = requests.get(fetch_url)

    if fetch_response.status_code != 200:
        print(f'\tStatus code = {fetch_response.status_code}')

    # Get the XML output with all the PMIDs and their abtracts
    pmid_data = fetch_response.text

    print('\tFunction get_abstracts complete.')

    return ids_of_interest,pmid_data

def get_data_from_xml(ids_of_interest:list,
                      pmid_data)->pd.DataFrame:
    '''
    Description: Extracts information from an XML string that has title, abstract, etc. for relevant papers and organizes into a df.
    Args:
        ids_of_interest = a list of PMIDs relevant to the actual query. Obtained from the get_abstracts function
        pmid_data = xml string containing all metadata and abtracts for multiple PMIDs, obtained from the get_abstracts function
    Returns: A df with information about each paper.
    '''

    # The XML string could have many html characters such as &#xa0 and &#x3ba and these may not be properly decoded by python
    # Clean up most of such strings using the html module

    print('\tPerforming basic cleanup.')
    cleaned_xml = html.unescape(pmid_data)

    # Check if html characters are cleaned up
    print(f'\t&#xa0 left: {cleaned_xml.count("&#xa0")}')
    print(f'\t&#x3ba left: {cleaned_xml.count("&#x3ba")}')
    print(f'\t&# left: {cleaned_xml.count("&#")}')

    data = []

    # Use the XML module and extract information from each paper
    root = ET.fromstring(pmid_data)

    print('Iterating through each article and collecting information.')
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

        # Adding a step to make sure PMIDs being extracted are relevant to search results.
        # This was added after noticing that within 1 paper, there could be multiple PMIDs present, but only 1 of them is paper-specific

        for count,pmid in enumerate(child.iter('PMID')):
            # For some papers, there could be multiple PMIDs associated with it.
            # The first one is the paper's PMID but the rest would be from other commentaries or papers that have highlighted the study.
            # So, the following code was added to only extract the first PMID
            if count==0 and pmid.text in ids_of_interest:
                # Add a step to check this PMID is relevant
                primary_id = pmid.text
                pmid_val += primary_id

                # Once primary ID is found then only look for other information:
                # This also helps in tracing incomplete info or problems specific to a PMID
                for title in child.iter('ArticleTitle'):
                    if title.text is not None:
                        title_val += title.text

                    else:
                        print(f'\t<FLAG> ArticleTitle is None for PMID {primary_id}')

                for abstract in child.iter('AbstractText'):

                    if abstract.get('Label','x') != 'x':
                        label = abstract.get('Label')
                        abstract_val += f'[{label}]'
                        if abstract.text is not None:
                            abstract_val += abstract.text
                    else:
                        if abstract.itertext() is not None:
                            abstract_val += ''.join(abstract.itertext())

                for journal in child.iter('Title'):
                    if journal.text is not None:
                        journal_val += journal.text

                for pubtype in child.iter('PublicationTypeList'):
                    if pubtype.itertext() is not None:
                        pubtype_val += '|'.join(pubtype.itertext())

                for pubdate in child.iter('PubDate'):
                    if pubdate.itertext() is not None:
                        pubdate_val += ' '.join(pubdate.itertext())

                for keyword in child.iter('KeywordList'):
                    if keyword.itertext() is not None:
                        keywords_val += '|'.join(keyword.itertext())

                # Collect everything from a single paper into a row
                row = {
                        'pmid':pmid_val,
                        'publication_date':pubdate_val,
                        'publication_type':pubtype_val,
                        'article_title':title_val,
                        'abstract':abstract_val,
                        'keywords':keywords_val,
                        'journal':journal_val
                }

                # Append the row
                data.append(row)

            else:
                print(f'\t<FLAG> Additional PMID {pmid.text} found in association with {primary_id}, not collected.')

    # Create a df
    papers_df = pd.DataFrame(data)

    print('\tFunction get_data_from_xml complete.')

    return papers_df

def run_pubmed_pipeline(query,
                        save_on_server,
                        search_format,
                        search_starting_index,
                        search_max_records,
                        sorting_criteria,
                        content_type,
                        fetch_starting_index,
                        fetch_max_records):

    print(f'----Running pipeline for the following query:----\n{query}')

    print('Using PubMed esearch API to get PMIDs matching the search query.')
    search_output = get_pmids(query=query,
                              save_on_server=save_on_server,
                              search_format=search_format,
                              search_starting_index=search_starting_index,
                              search_max_records=search_max_records,
                              sorting_criteria=sorting_criteria)

    print('Collecting metadata about the search results into a dictionary.')
    metadata_dict = query_and_ids(search_output)

    print('Using PubMed efetch API to get abstract and other details for relevant PMIDs into an XML string.')
    if save_on_server=='y':
        ids_from_server=True
    else:
        ids_from_server = False

    ids_of_interest,pmid_data = get_abstracts(search_output=search_output,
                                              content_type=content_type,
                                              ids_from_server=ids_from_server,
                                              fetch_starting_index=fetch_starting_index,
                                              fetch_max_records=fetch_max_records)

    print('Extracting data from XML string and organizing it into a dataframe.')
    papers_df = get_data_from_xml(ids_of_interest,
                                  pmid_data)

    # Collect info about actually how many abstracts were collected
    num_retrieved_papers = len(set(papers_df['pmid'].values))
    papers_df['num_abstracts_retrieved'] = num_retrieved_papers

    num_abstracts_requested = len(ids_of_interest)
    papers_df['num_abstracts_requested'] = num_abstracts_requested

    print('Adding in metadata information.')
    for col,val in metadata_dict.items():
        papers_df[col] = val

    print('Pipeline complete, dataframe ready.')
    return papers_df
