'''
Overview
'''

# imports
import numpy as np
import pandas as pd
import string
from fuzzywuzzy import fuzz

def get_pmids_without_entity(df,
                             entity_column_name:str):


    pmids = df.groupby('pmid')[entity_column_name].nunique()[lambda x: x==0].index

    return pmids

def get_drug_metrics(model_df,
                     gtruth_df,
                     column_name:str,
                     score_cutoff:int):
    '''
    Description:
    Args:
        model_df = df containing results from the GPT completions
        gtruth_df = df containing manually curated ground truth data
        scorer = the type of fuzzy match scorer to use. eg. token_set_ratio or token_sort_ratio
        score_cutoff = int,a score greater than or equal to this cutoff will mean two strings are a match using the defined scorer
                       highest = 100
    Returns:
        metrics = dictionary with all the confusion matrix values for: tp,tn,fp,fn
        pmids_errors = a dictionary showing which PMIDs contain false positive or false negative errors
        matches = a list of dictionaries showing which drug name from the ground truth data and the model matched for each PMID
    '''
    # Keep count of the metrics
    metrics = {'tp':0,'fp':0,'fn':0, 'tn':0}

    # Dictionary to ID where the errors are; list will have PMIDs
    pmids_errors = {'fp':[],'fn':[]}

    # Get all PMIDs
    all_pmids = set(model_df['pmid'].unique())

    # get ground truth papers that have no drugs in it
    gtruth_no_drugs = get_pmids_without_entity(gtruth_df,entity_column_name=column_name)

    # get pmids for which the model didn't extract any drugs
    model_no_drugs = get_pmids_without_entity(model_df,entity_column_name=column_name)

    # Get pmids that intersect between groundtruth and model results, where no drug is present
    # These are correctly processed to have no drug names, so these are the true negatives
    common_no_drugs = set(gtruth_no_drugs).intersection(set(model_no_drugs))
    metrics['tn'] += len(common_no_drugs)

    # Remove these common pmids from consideration into the next loop
    compare_pmids = all_pmids.difference(common_no_drugs)

    # Collect the drugs that matched between the model and ground truth
    matches = []

    # For each pmid, compare the extracted drug names to those in the ground truth data
    for pmid in compare_pmids:
        # Keep track of which drugs in manually curated data have been matched for a PMID
        gtruth_matched = set()

        # Get drugs extracted by the model for this paper
        model_drugs = model_df[model_df['pmid']==pmid][column_name].unique().tolist()

        # Get drugs present in ground truth curate data for this paper
        gtruth_drugs = gtruth_df[gtruth_df['pmid']==pmid][column_name].unique().tolist()

        # Now take each drug extracted by the model and compare it to see if it matches each of the drugs present in the ground truth list
        for model_drug in model_drugs:
            # Initialize a variable to track matches
            match = False
            for gtruth_drug in gtruth_drugs:

                # Get fuzzy matching score after removing special characters as these will affect the score
                cleaned_model_drug = model_drug.translate(str.maketrans('', '', string.punctuation))
                cleaned_gtruth_drug = gtruth_drug.translate(str.maketrans('', '', string.punctuation))

                score = fuzz.token_set_ratio(cleaned_model_drug,cleaned_gtruth_drug)

                # Stop searching for a match with ground truth drugs once a match is found.
                '''Adding this break ensures that if there are 2 similar analogs of a drug in the gound truth set
                and the model extracts only 1, then that difference will be accounted here.'''
                if score >= score_cutoff:
                    match = True
                    # Collect the matched pairs
                    matches.append({'pmid':pmid,
                                   'model_drug_name':model_drug,
                                   'gtruth_drug_name':gtruth_drug})
                    gtruth_matched.add(gtruth_drug)
                    gtruth_drugs.remove(gtruth_drug)
                    break

            if match is True:
                metrics['tp'] += 1
            else:
                metrics['fp'] += 1
                pmids_errors['fp'].append(pmid)
                print(f'PMID {pmid} has a false positive.')

        # Whichever drugs from ground truth that didn't get matched at all are false negatives that the model failed to extract
        fn_drugs = set(gtruth_drugs).difference(gtruth_matched)

        # Remove any None values; these will still count towards the length
        fn_drugs_filtered = {x for x in fn_drugs if x is not None}

        if len(fn_drugs_filtered) > 0:
            # If false negative are found, collect the PMID for tracing the errors later
            pmids_errors['fn'].append(pmid)
            print(f'PMID {pmid}: {len(fn_drugs_filtered)} false negatives.')
        metrics['fn'] += len(fn_drugs_filtered)

    return metrics, pmids_errors, matches

def get_target_metrics(model_df,
                       gtruth_df,
                       matches:list,
                       score_cutoff:int):
    '''
    Description:
    Args:
        model_df = df containing results from the GPT completions
        gtruth_df = df containing manually curated ground truth data
        matches = a list of dictionaries showing which drug name from the ground truth data and the model matched for each PMID
                  output of get_drug_metrics function
        score_cutoff = int,a score greater than or equal to this cutoff will mean two strings are a match using the defined scorer
                       highest = 100

    Returns:
    '''
    # Dictionary to ID where the errors are; list will have PMIDs
    targets_pmids_errors = {'fp':[],'fn':[]}
    target_metrics = {'fp':0,'fn':0}

    # Collect matches - pmid,drug,drug-target
    target_matches = []

    # Iterate over matched drugs and compare associated targets
    for matched_drug in matches:
        # Get targets
        model_targets = model_df[(model_df['pmid']==matched_drug['pmid']) & (model_df['drug_name']==matched_drug['model_drug_name'])]['direct_target'].unique().tolist()

        gtruth_targets = gtruth_df[(gtruth_df['pmid']==matched_drug['pmid']) & (gtruth_df['drug_name']==matched_drug['gtruth_drug_name'])]['direct_target'].unique().tolist()

        # Compare the targets using token_sort_ratio. For target names, a stricter cutoff is used.
        for model_target in model_targets:

            for gtruth_target in gtruth_targets:

                # Get fuzzy matching score after removing special characters as these will affect the score
                # None values will throw an error, so variables are first converted to a string
                cleaned_model_target = str(model_target).translate(str.maketrans('', '', string.punctuation))
                cleaned_gtruth_target = str(gtruth_target).translate(str.maketrans('', '', string.punctuation))

                # Get a score
                score = fuzz.token_sort_ratio(cleaned_model_target,cleaned_gtruth_target)

                # If there is match, collect it as a dictionary
                if score >= score_cutoff:
                    matched_drug['model_direct_target'] = model_target
                    matched_drug['gtruth_direct_target'] = gtruth_target

                    # Collect the match
                    target_matches.append(matched_drug)

                    # Remove the matched target and stop looking for a match
                    gtruth_targets.remove(gtruth_target)
                    model_targets.remove(model_target)
                    break

        # If a model target has not been matched to anything in ground truth, it will be a false positive
        target_metrics['fp'] += len(model_targets)
        # Add the PMID to trace errors
        if len(model_targets) > 0:
            targets_pmids_errors['fp'].append(matched_drug['pmid'])

        # Whatever ground truth targets are left were missed by the model and hence are false negatives
        target_metrics['fn'] += len(gtruth_targets)
        if len(gtruth_targets) > 0:
            targets_pmids_errors['fn'].append(matched_drug['pmid'])

    # Get true positive and true negative numbers
    target_matches_df = pd.DataFrame(target_matches)

    # Number of rows with None will be all true negative drug-target pairs
    tn = len(target_matches_df[target_matches_df['model_direct_target'].isna() & target_matches_df['gtruth_direct_target'].isna()])

    # Remaining rows will be true positive drug-target pairs
    tp = len(target_matches_df[~(target_matches_df['model_direct_target'].isna() & target_matches_df['gtruth_direct_target'].isna())])

    # Add these numbers
    target_metrics['tp'] = tp
    target_metrics['tn'] = tn

    return target_metrics, targets_pmids_errors, target_matches

def get_interaction_metrics(model_df,
                            gtruth_df,
                            target_matches:list,
                            score_cutoff:int):
    '''
    Description:
    Args:
        model_df = df containing results from the GPT completions
        gtruth_df = df containing manually curated ground truth data
        target_matches = a list of dictionaries showing which drug and target pairs from ground truth data and the model matched for each PMID
                         output of get_drug_metrics function
        score_cutoff = int,a score greater than or equal to this cutoff will mean two strings are a match using the defined scorer
                       highest = 100
    Returns:
    '''
    # Dictionary to ID where the errors are; list will have PMIDs
    int_pmids_errors = {'fp':[],'fn':[]}
    int_metrics = {'fp':0,'fn':0}

    # Collect matches - pmid,drug,target, and interaction
    int_matches = []
    # Iterate over matched drugs and compare associated targets
    for matched_pair in target_matches:
        df = pd.DataFrame([matched_pair])

        # Extract matching rows and then get interaction type value
        model_ints = pd.merge(df[['pmid','model_drug_name','model_direct_target']],
                              model_df,
                              how='inner',
                              left_on=['pmid','model_drug_name','model_direct_target'],
                              right_on=['pmid','drug_name','direct_target'])['drug-direct_target_interaction'].unique().tolist()

        gtruth_ints = pd.merge(df[['pmid','gtruth_drug_name','gtruth_direct_target']],
                              gtruth_df,
                              how='inner',
                              left_on=['pmid','gtruth_drug_name','gtruth_direct_target'],
                              right_on=['pmid','drug_name','direct_target'])['drug-direct_target_interaction'].unique().tolist()

        # Compare the interaction using token_set_ratio.
        for model_int in model_ints:

            for gtruth_int in gtruth_ints:
                # Get fuzzy matching score after removing special characters as these will affect the score
                # None values will throw an error, so variables are first converted to a string
                cleaned_model_int = str(model_int).translate(str.maketrans('', '', string.punctuation))
                cleaned_gtruth_int = str(gtruth_int).translate(str.maketrans('', '', string.punctuation))
                # Get a score
                score = fuzz.token_set_ratio(cleaned_model_int,cleaned_gtruth_int)

                # If there is match, collect it as a dictionary
                if score >= score_cutoff:
                    matched_pair['model_drug-direct_target_interaction'] = model_int
                    matched_pair['gtruth_drug-direct_target_interaction'] = gtruth_int

                    # Collect the match
                    int_matches.append(matched_pair)

                    # Remove the matched target and stop looking for a match
                    gtruth_ints.remove(gtruth_int)
                    model_ints.remove(model_int)
                    break

        # If a model interaction has not been matched to anything in ground truth, it will be a false positive
        int_metrics['fp'] += len(model_ints)
        # Add the PMID to trace errors
        if len(model_ints) > 0:
            int_pmids_errors['fp'].append(matched_pair['pmid'])

        # Whatever ground truth interaction is left was missed by the model and hence is false negative
        int_metrics['fn'] += len(gtruth_ints)
        if len(gtruth_ints) > 0:
            int_pmids_errors['fn'].append(matched_pair['pmid'])

    # Get true positive and true negative numbers
    int_matches_df = pd.DataFrame(int_matches)

    # Number of rows with None will be all true negative drug-target pairs
    tn = len(int_matches_df[int_matches_df['model_drug-direct_target_interaction'].isna() & int_matches_df['gtruth_drug-direct_target_interaction'].isna()])

    # Remaining rows will be true positive drug-target pairs
    tp = len(int_matches_df[~(int_matches_df['model_drug-direct_target_interaction'].isna() & int_matches_df['gtruth_drug-direct_target_interaction'].isna())])

    # Add these numbers
    int_metrics['tp'] = tp
    int_metrics['tn'] = tn

    return int_metrics, int_pmids_errors, int_matches

def calculate_metrics(metrics:dict):
    '''
    Description:
    Args:
        metrics = dictionary with all the confusion matrix values for: tp,tn,fp,fn
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
