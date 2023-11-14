#!/usr/bin/env python
# authors: Lisa Sequeira, Althrun Sun, Luke Yang, Edward Zhang
# date: 2023-06-21

"""This script is the pipeline for testing the similarity
Usage: annoy_model.py --array_path=<array_path> --output_model_path=<output_model_path> 
Options:
--array_path=<array_path> This is path for the vectorized array
--output_model_path=<output_model_path> This is path for the output ANNOY model
"""
from docopt import docopt
from annoy import AnnoyIndex
import numpy as np

def load_arrays(array_path: str):   
    """
    Load arrays from the specified path.

    Parameters:
    array_path (str): Path to the directory containing the arrays.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing three arrays: X, Z, and dummy.

    """
    with open(array_path + 'X.npy', 'rb') as f:
        X = np.load(f)
    with open(array_path + 'Z.npy', 'rb') as f:
        Z = np.load(f) 
    with open(array_path + 'dummy.npy', 'rb') as f:
        dummy = np.load(f)
    return X, Z, dummy

def build_annoy_model(transformed_df:np.ndarray, n_trees: int = 10, metric: str = 'angular'):
    """
    Build an Annoy model using the transformed data.

    Parameters:
    transformed_df (np.ndarray): Transformed data as a NumPy array.
    n_trees (int): Number of trees to build in the Annoy model. Default is 10.
    metric (str): Metric to use for similarity calculation in the Annoy model. Default is 'angular'.

    Returns:
    AnnoyIndex: Annoy model built using the transformed data.

    """
    t = AnnoyIndex(transformed_df.shape[1], metric)
    
    for i in range(len(transformed_df)):
        item_vector = transformed_df[i]
        t.add_item(i, item_vector)
    
    t.build(n_trees)
    
    return t

def build_and_save_annoy_models(X:np.ndarray,
                                Z:np.ndarray,
                                dummy:np.ndarray,
                                output_model_path):
    """
    Build and save Annoy models based on the given arrays.

    Parameters:
    X (np.ndarray): Array X for building the first Annoy model.
    Z (np.ndarray): Array Z for building the second Annoy model.
    dummy (np.ndarray): Dummy array for building the control Annoy model.
    output_model_path (str): Path to save the Annoy models.

    """

    a_1 = build_annoy_model(X)
    a_2 = build_annoy_model(Z)

    a_1.save(output_model_path + 'a_1.ann')
    a_2.save(output_model_path + 'a_2.ann')

    if dummy is not None:
        control = build_annoy_model(dummy)
        control.save(output_model_path + 'baseline.ann')
    
def main():
    opt = docopt(__doc__)
    array_path = opt["--array_path"]
    output_model_path = opt["--output_model_path"]
    print("Building and saving the ANNOY model.")
    X, Z, dummy = load_arrays(array_path)
    build_and_save_annoy_models(X, Z, dummy, output_model_path)


    


if __name__ == "__main__":
    main()