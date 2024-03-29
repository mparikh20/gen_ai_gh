'''
Overview
'''

# imports
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz

def get_pmids_without_entity(df,
                             entity_column_name:str):


    indices = np.where(df.groupby('pmid')[entity_column_name].nunique() ==0)

    pmids = df.loc[indices]['pmid'].unique()

    return pmids

def get_drug_metrics(model_df,
                     gtruth_df,
                     column_name):
    '''
    Description:
    Args:
    Returns:
    '''
    metrics = {'tp':0,'fp':0,'fn':0, 'tn':0}

    all_pmids = set(model_df['pmid'].unique())

    # get ground truth papers that have no drugs in it
    gtruth_no_drugs = get_pmids_without_entity(gtruth_df,entity_column_name='drug_name')

    # get pmids for which the model didn't extract any drugs
    model_no_drugs = get_pmids_without_entity(model_df,entity_column_name='drug_name')

    # Get pmids that intersect between groundtruth and model results, where no drug is present
    # These are correctly processed to have no drug names, so these are the true negatives
    common_no_drugs = set(gtruth_no_drugs).intersection(set(model_no_drugs))
    metrics['tn'] += len(common_no_drugs)

    # Remove these common pmids from consideration into the next loop
    compare_pmids = all_pmids.difference(common_no_drugs)

    # For each pmid, compare the extracted drug names to those in the ground truth data
    for pmid in compare_pmids:
        # Keep track of which drugs in manually curated data have been matched for a PMID
        gtruth_matched = set()

        # Get drugs extracted by the model for this paper
        model_drugs = model_df[model_df['pmid']==pmid]['drug_name'].unique()

        # Get drugs present in ground truth curate data for this paper
        gtruth_drugs = gtruth_df[gtruth_df['pmid']==pmid]['drug_name'].unique()

        # Now take each drug extracted by the model and compare it to see if it matches each of the drugs present in the ground truth list
        for model_drug in model_drugs:
            # Initialize a variable to track matches
            match = False
            for gtruth_drug in gtruth_drugs:
                # Get fuzzy matching score
                score = fuzz.token_set_ratio(model_drug,gtruth_drug)

                if score == 100:
                    match = True
                    gtruth_matched.add(gtruth_drug)

            if match == True:
                metrics['tp'] += 1
            else:
                metrics['fp'] += 1

        # Whichever drugs from ground truth that didn't get matched at all are false negatives that the model failed to extract
        fn_drugs = set(gtruth_drugs).difference(gtruth_matched)
        metrics['fn'] += len(fn_drugs)

    return metrics

