#!/usr/bin/env python
# authors: Lisa Sequeira, Althrun Sun, Luke Yang, Edward Zhang
# date: 2023-06-21

"""This script generates the similarity using known data
Usage: sim_model_validation.py --original_df_path=<original_df_path> --validation_df_path=<validation_df_path> --array_path=<array_path> --model_path=<model_path> --output_path=<output_path> 
Options:
--original_df_path=<original_df_path> This is path for the original dataframe
--validation_df_path=<validation_df_path> This is path for the known similar data
--model_path=<model_path> This is path for the Annoy models
--output_path=<output_path> This is path for the output similarity dataframe
"""
from docopt import docopt
import pandas as pd
import numpy as np
from annoy import AnnoyIndex
from collections import defaultdict

def get_title(idx, df):
    """
    Gets title given an index and dataframe.

    Parameters:
    idx (int): The index value.
    df (pandas.DataFrame): Input DataFrame with title column.

    Returns:
    str: The title value.
    """
    return df['Title'][int(idx)]


def get_product_id(idx, df):
    """
    Gets product given an index and dataframe.

    Parameters:
    idx (int): The index value.
    df (pandas.DataFrame): Input DataFrame with idProduct column.

    Returns:
    str: The ProductID value.
    """
    return df['idProduct'][int(idx)]


def compare_model(furniture_df, item_pairs: list, models: dict):
    """
    Creates a dataframe containing item_pairs found in furniture_df and the similarity 
    scores generated by each of the models. 

    Parameters:
    furniture_df (pandas.DataFrame): The DataFrame containing the item pairs and product 
    titles. It should be the same df that was used to train the models. 
    models (ANNOY models): Input dictionary of ANNOY models. 

    Returns:
    pandas.DataFrame: The outputed dataframe.
    """
    df_1 = pd.DataFrame(item_pairs, columns=["Item_1_Index", "Item_2_Index"])

    similarity_scores = defaultdict(list)

    for model in models:
        for x, y in item_pairs:
            score = models[model].get_distance(x, y)
            similarity_scores[model + '_Score'].append(1 - score)

    df_2 = pd.DataFrame(similarity_scores)

    concat_df = pd.concat([df_1, df_2], axis="columns")

    try:
        furniture_df['Title']
    except KeyError:
        raise ValueError("Missing 'Title' column in furniture_df")
    
    try:
        furniture_df['idProduct']
    except KeyError:
        raise ValueError("Missing 'iDProduct' column in furniture_df")
    

    concat_df.insert(1, 'Item_1_ProductID', concat_df['Item_1_Index'].apply(
        get_product_id, df=furniture_df))
    concat_df.insert(2, 'Item_1_Title', concat_df['Item_1_Index'].apply(
        get_title, df=furniture_df))
    concat_df.insert(3, 'Item_2_ProductID', concat_df['Item_2_Index'].apply(
        get_product_id, df=furniture_df))
    concat_df.insert(4, 'Item_2_Title', concat_df['Item_2_Index'].apply(
        get_title, df=furniture_df))
    concat_df.drop(["Item_1_Index", "Item_2_Index"], axis=1, inplace=True)

    return concat_df

def build_validation_list(df,validation_df_path):
    """
    Creates a list containing similar product item pairs for validation. 

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the all products details 
    and productID.  
    validation_df_path (str): Path to dataframe containing productID and 
    grouping details.  

    Returns:
    list: The outputed list of similar items. 
    """

    df_grouping = pd.read_csv(validation_df_path)

    try:
        df_grouping['idProduct']
    except KeyError:
        raise ValueError("Missing 'idProduct' column from data found in validation_df_path")
    
    try:
        df['idProduct']
    except KeyError:
        raise ValueError("Missing 'iDProduct' column in df")

    df_merge = df.merge(df_grouping, on="idProduct", how='inner')

    try:
        df_merge['ProductGroupID']
    except KeyError:
        raise ValueError("Missing 'ProductGroupID' column in data")

    #Keep only groups that have more than one furniture items in it
    duplicates = df_merge["ProductGroupID"].duplicated(keep=False)
    df_group_final = df_merge[duplicates].sort_values(
        by = 'ProductGroupID'
        ).drop_duplicates(
            subset=["idProduct"])
    item_value_1 = 0
    item_value_2 = 0
    item_pair = []
    similiar_items = []

    for item_idx in df_group_final["Unnamed: 0"]:
        
        if item_value_1 == 0:
            item_value_1 = item_idx
        else:
            item_value_2 = item_idx
            item_pair = [item_value_1, item_value_2]
            item_value_1 = 0
            item_value_2 = 0

        if len(item_pair) == 2:
            similiar_items.append(item_pair)
            item_pair = []
    return similiar_items

def load_models(model_path:str, array_path:str):
    """
    Loads saved ANNOY models from model_path on transformed arrays in array_path. 

    Parameters:
    model_path (str): Path to the saved and fitted ANNOY models.
    array_path (numpy.array): Path to transformed arrays used to train the models

    Returns:
    control: The ANNOY model without entities. 
    a_1: The ANNOY model with entities & all other features
    a_2: The ANNOY model with entities only.
    """

    with open(array_path + 'X.npy', 'rb') as f:
        X = np.load(f)
    with open(array_path + 'Z.npy', 'rb') as f:
        Z = np.load(f)
    with open(array_path + 'dummy.npy', 'rb') as f:
        dummy = np.load(f)

    X_width = X.shape[1]
    Z_width = Z.shape[1]
    dummy_width = dummy.shape[1]
    control = AnnoyIndex(dummy_width, 'angular')
    a_1 = AnnoyIndex(X_width, 'angular')
    a_2 = AnnoyIndex(Z_width, 'angular')
    control.load(model_path+'baseline.ann')
    a_1.load(model_path+'a_1.ann')
    a_2.load(model_path+'a_2.ann')

    return control, a_1, a_2




def main():
    opt = docopt(__doc__)
    original_df_path = opt["--original_df_path"]
    validation_df_path = opt["--validation_df_path"]
    array_path = opt["--array_path"]
    model_path = opt["--model_path"]
    output_path = opt["--output_path"]

    df = pd.read_csv(original_df_path)
    print("Generating the similarity validation list.")
    positive_lst = build_validation_list(df, validation_df_path)
    control, a_1, a_2 = load_models(model_path, array_path)
    annoy_models = {'No_NER_Baseline':control,'All_Features_ANNOY': a_1, 'NER_only_ANNOY': a_2}
    df_similar = compare_model(df, positive_lst, annoy_models)
    df_similar.to_csv(output_path)


if __name__ == "__main__":
    main()
