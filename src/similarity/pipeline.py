#!/usr/bin/env python
# authors: Lisa Sequeira, Althrun Sun, Luke Yang, Edward Zhang
# date: 2023-06-21

"""This script is the pipeline for testing the similarity
Usage: pipeline.py --base_dataframe=<base_dataframe> 
Options:
--base_dataframe=<base_dataframe> This is dataframe you want to find the similarity on
"""
from docopt import docopt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
import gensim.downloader as api
import torch
import pickle
import numpy as np

#Model used for Sentence trasnformation
#Can be swapped with another transformer model
model_bert = SentenceTransformer('bert-base-nli-mean-tokens')

#Enter your choice of columns for each category
#Used in pipeline code below
SENT_TRANSFORM = [
    'STYLE_sentence', 
    'APPEARANCE_sentence', 
    'FEATURE_sentence',
    'NOTICE_sentence',
    'COLOR_sentence',
    'TYPE_sentence',
    'ADDITIONAL_MATERIAL_sentence',
    'PrimaryMaterial_sentence']

NUM_TRANSFORM = [
    'price', 
    'weight', 
    'width', 
    'height', 
    'depth']

BIN_TRANSFORM = ['onPromo']

SENT_TRANSFORM_2 = [
    'STYLE_sentence', 
    'APPEARANCE_sentence', 
    'FEATURE_sentence',
    'NOTICE_sentence',
    'COLOR_sentence',
    'TYPE_sentence',
    'ADDITIONAL_MATERIAL_sentence']

BASELINE_SENT_TRANSFORM = [
    'combinedDescription', 
    'PrimaryMaterial']

def bert_encoding(sent, model = model_bert):   
    """
    Encode the given sentence using the BERT model.

    Parameters:
    sent (str): The input sentence to be encoded.
    model (BertModel): The BERT model used for encoding. Default is `model_bert`.

    Returns:
    np.array: Numpy array of the encoded sentence.

    """ 
    embeddings = model.encode(sent)  
    return np.array(embeddings)

def build_pipelines(sent_features:list, 
                    bin_features:list = [],
                    num_features:list = [] ):
    """
    Build and configure the preprocessing pipelines.

    Returns:
    Preprocessor (ColumnTransformer): Preprocessing pipeline for the model.

    """
    sent_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='')), 
    ('bert', FunctionTransformer(bert_encoding, validate=False))
    ])
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    bin_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]) 

    if not bin_features and not num_features:
        preprocessor = ColumnTransformer(
        transformers=[
            ('sent_features',sent_transformer, sent_features)
            ]
        )
    elif not bin_features:
        preprocessor = ColumnTransformer(
        transformers=[
            ('sent_features',sent_transformer, sent_features),
            ('num', num_transformer, num_features)
            ]
        )
    elif not num_features:
        preprocessor = ColumnTransformer(
        transformers=[
            ('sent_features',sent_transformer, sent_features),
            ('binary', bin_transformer, bin_features)
            ]
        )
    else:
        preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('binary', bin_transformer, bin_features),
            ('sent_features',sent_transformer, sent_features)
            ]
        )

    return preprocessor

def fit_transform_save(preprocessor, pipe_file_name, array_file_name, df):
    """
    Fit the preprocessor on the given DataFrame, save the pipeline and transformed array.

    Parameters:
    preprocessor (ColumnTransformer): Preprocessing pipeline to be fitted.
    pipe_file_name (str): Name of the file to save the pipeline.
    array_file_name (str): Name of the file to save the transformed array.
    df (pd.DataFrame): Input DataFrame for fitting the preprocessor.

    Returns:
    None

    """
    preprocessor.fit(df)

    with open('models/similarity_model/pipeline-models/' + pipe_file_name + '.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    trans_array = preprocessor.transform(df)
    
    with open('data/interim/vect_emb_pipelines/' + array_file_name + '.npy', 'wb') as f:
        np.save(f, trans_array)


def main():
    opt = docopt(__doc__)
    base_dataframe = opt["--base_dataframe"]
    print("Initializing the similarity pipeline.")
    print("Loading data frame...")
    df = pd.read_csv(base_dataframe)
    control_preprocessor = build_pipelines(BASELINE_SENT_TRANSFORM,
                                           BIN_TRANSFORM,
                                           NUM_TRANSFORM)
    preprocessor = build_pipelines(SENT_TRANSFORM,
                                   BIN_TRANSFORM,
                                   NUM_TRANSFORM)
    preprocessor2 = build_pipelines(SENT_TRANSFORM_2)
    print("Fitting the entire dataframe using the preprocessor")
    fit_transform_save(preprocessor,'preprocessor', 'X', df)
    print("Fitting the NER only preprocessor")
    fit_transform_save(preprocessor2,'preprocessor2', 'Z', df)
    if torch.has_cuda:
        print("Fitting the baseline preprocessor")
        fit_transform_save(control_preprocessor,'baseline', 'dummy', df)


if __name__ == "__main__":
    main()