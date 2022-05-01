import time, os
import json
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# DATATHON files constants
DATA_DIR = "data"

RAW_DATA =  os.path.join(DATA_DIR, "labels_racism.csv")
RAW_PREPROC_DATA =  os.path.join(DATA_DIR, "labels_racism_preproc.csv") # with IDs and labels
AGGREGATED_DATA = os.path.join(DATA_DIR, "labels_racism_aggregated.csv")
EVAL_SAMPLE = os.path.join(DATA_DIR, "evaluation_sample.csv")
EVAL_PUBLIC = os.path.join(DATA_DIR, "evaluation_public.csv")
IDS_VALIDATION_SET = os.path.join(DATA_DIR, "ids_validation_set.json")


def load_dataset(filename=RAW_DATA):
    """
    Import df from path
    filename: String.
    """
    df = pd.read_csv(filename, sep='|')

    # If aggregated, load as list 
    if filename == AGGREGATED_DATA:
        list_cols = ["labeller_id", "label", "ind_value"]
        for list_col in list_cols:
            df[list_col] = df[list_col].apply(lambda x: literal_eval(x))
            
    return df


def binarize_label(df, col, thr=0.5, strict=False, inplace=False):
    """
    Binarize to categorical labels
    df: pd.DataFrame.
    col: String. Name of the column with values [0.0,1.0] to binarize
    thr: np.int64. Threshold value (“racist” if higher, “non-racist” if lower)
    strict: Bool. Labels for strict/non-strict classifier.

    Usage example to generate labels for non strict classifier at 0.35:
        binarize_label(aggregated, "w_m_vote", thr=0.35, strict=False, inplace=True)
        aggregated.loc[aggregated.w_m_vote <0.5,].sample(10, random_state=RANDOM_STATE)
    """

    if strict:
        condition = df[col]>thr
    else:
        condition = df[col]>=thr

    if inplace:
        print("Saving as a new {}_label column".format(col))
        df["{}_label".format(col)] = np.where(condition, "racist", "non-racist")
        return df
    else:
        return np.where(condition, "racist", "non-racist")

def f1(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label='racist')
    # same as:
    # target_names = ['non-racist', 'racist']
    # y_true = [target_names.index(el) for el in y_true]
    # y_pred = [target_names.index(el) for el in y_pred]
    # return f1_score(y_true, y_pred)


def get_validation_set(filename=AGGREGATED_DATA, test_size=0.1, random_seed=42):
    if not os.path.isfile(IDS_VALIDATION_SET):
        # df_splitted_train, df_splitted_test = train_test_split(aggregated_df, stratify=aggregated_df['w_m_vote_label'], test_size=test_size, random_state=random_seed)
        # recomputing this on a different python version/environment gives different results, so just throw an error
        raise ValueError(f'file {IDS_VALIDATION_SET} does not exist')
    ids_validation = read_json(IDS_VALIDATION_SET)
    aggregated_df = load_dataset(filename)
    if filename in [RAW_PREPROC_DATA, RAW_DATA]:
        # the raw data has still unknowns
        aggregated_df = aggregated_df[aggregated_df.label != 'unknown']
    df_splitted_test = aggregated_df[aggregated_df['id'].isin(ids_validation)]
    return df_splitted_test

def read_json(path: str):
    with open(path) as f:
        return json.load(f)