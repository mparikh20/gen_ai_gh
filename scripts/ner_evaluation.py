'''
Overview
'''

# imports
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz

def get_pmids_without_entity(df,
                             entity_column_name:str):


    pmids = df.groupby('pmid')[entity_column_name].nunique()[lambda x: x==0].index

    return pmids

def get_drug_metrics(model_df,
                     gtruth_df,
                     column_name):
    '''
    Description:
    Args:
    Returns:
    '''
    # Keep count of the metrics
    metrics = {'tp':0,'fp':0,'fn':0, 'tn':0}

    # Dictionary to ID where the errors are; list will PMIDs
    pmids_errors = {'fp':[],'fn':[]}

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
        model_drugs = model_df[model_df['pmid']==pmid]['drug_name'].unique().tolist()

        # Get drugs present in ground truth curate data for this paper
        gtruth_drugs = gtruth_df[gtruth_df['pmid']==pmid]['drug_name'].unique().tolist()

        # Now take each drug extracted by the model and compare it to see if it matches each of the drugs present in the ground truth list
        for model_drug in model_drugs:
            # Initialize a variable to track matches
            match = False
            for gtruth_drug in gtruth_drugs:
                # Get fuzzy matching score
                score = fuzz.token_set_ratio(model_drug,gtruth_drug)

                # Stop searching for a match with ground truth drugs once a match is found.
                '''Adding this break ensures that if there are 2 similar analogs of a drug in the gound truth set
                and the model extracts only 1, then that difference will be accounted here.'''
                if score == 100:
                    match = True
                    gtruth_matched.add(gtruth_drug)
                    gtruth_drugs.remove(gtruth_drug)
                    break

            if match is True:
                metrics['tp'] += 1
            else:
                metrics['fp'] += 1
                pmids_errors['fp'].append(pmid)

        # Whichever drugs from ground truth that didn't get matched at all are false negatives that the model failed to extract
        fn_drugs = set(gtruth_drugs).difference(gtruth_matched)

        # Remove any None values; these will still count towards the length
        fn_drugs_filtered = {x for x in fn_drugs if x is not None}

        if len(fn_drugs_filtered) > 0:
            # If false negative are found, collect the PMID for tracing the errors later
            pmids_errors['fn'].append(pmid)
            print(f'PMID {pmid}: {len(fn_drugs_filtered)} false negatives.')
        metrics['fn'] += len(fn_drugs_filtered)

    return metrics, pmids_errors

def calculate_metrics(metrics:dict):
    '''
    Description:
    Args:
    Returns:
    '''
    eval_metrics = {}
    accuracy = (metrics['tp']+metrics['tn']) / (metrics['tp'] + metrics['tn'] + metrics['fp'] + metrics['fn'])

    # Calculate precision
    precision = metrics['tp'] / (metrics['tp'] + metrics['fp'])

    # Calculate recall
    recall = metrics['tp'] / (metrics['tp'] + metrics['fn'])

    f1_score = 2 * ((precision*recall) / (precision+recall))

    neg_pred_value = metrics['tn'] / (metrics['tn']+metrics['fn'])

    eval_metrics['accuracy']=round(accuracy,2)
    eval_metrics['precision']=round(precision,2)
    eval_metrics['recall']=round(recall,2)
    eval_metrics['f1_score']=round(f1_score,2)
    eval_metrics['neg_pred_value']=round(neg_pred_value,2)

    return eval_metrics
