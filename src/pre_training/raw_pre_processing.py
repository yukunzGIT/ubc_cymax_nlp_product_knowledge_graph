#!/usr/bin/env python
# authors: Lisa Sequeira, Althrun Sun, Luke Yang, Edward Zhang
# date: 2023-06-07

"""This script loads and preprocesses the data and exports to processed data folder.

Usage: raw_pre_processing.py --raw_path=<raw_path> --output_path=<output_path> [--exclude_sets=<exclude_sets>]
Options:
--raw_path=<raw_path>         This is the path to the raw data
--output_path=<output_path>   This is the path to where the processed data should be saved
--exclude_sets=<exclude_sets> Whether the user wants to include only single items or include furniture sets
"""

from docopt import docopt
from typing import Union, List
import pandas as pd
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from collections import Counter
import re

nltk.download('punkt')

def tokenize_text(text: str) -> List[str]:
    """
    Tokenize the given text by splitting on '/' and then using word_tokenize.

    Parameters:
        text (str): The input text to tokenize.

    Returns:
        List[str]: The list of tokens obtained from the text.
    """
    tokens = text.split("/")
    text = " ".join(tokens)
    tokens = word_tokenize(text)
    return tokens


def tokens_to_str(tokens: List[str]) -> str:
    """
    Convert a list of tokens to a string by joining them with spaces.

    Parameters:
        tokens (List[str]): The list of tokens to convert.

    Returns:
        str: The string representation of the tokens.
    """
    return ' '.join(tokens)


def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags from the given text using regular expressions.

    Parameters:
        text (str): The input text containing HTML tags.

    Returns:
        str: The text with HTML tags removed.
    """
    try:
        clean = re.split('<.*?>', text)
    except:
        return ' '
    return ' '.join(clean)

def not_contain_set(text: str) -> bool:
    """
    Check if the given text does not contain the word 'set' (case-insensitive).

    Parameters:
        text (str): The text to check.

    Returns:
        bool: True if the text does not contain 'set', False otherwise.
    """
    return not text.lower().__contains__('set')


def exclude_set_items(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exclude rows from the DataFrame that contain the word 'set' in specific columns.

    Parameters:
        df (pd.DataFrame): The DataFrame to filter.

    Returns:
        pd.DataFrame: The filtered DataFrame with rows excluded.
    """
    df = df[df['DescAccCategoryLevel2'].apply(not_contain_set)]
    df = df[df['DescCategoryLevel5'].apply(not_contain_set)]
    df = df[df['combinedDescription'].apply(not_contain_set)]
    return df

def load_and_preprocess_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess raw data from a CSV file.

    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The preprocessed DataFrame containing the loaded data.
    """
    df = pd.read_csv(file_path)
    print("Removing HTML tags...")
    df['details'] = df['details'].apply(remove_html_tags)
    df['combinedDescription'] = df['Title'] + '. Description: ' + df['details']
    print("Tokenizing raw data...")
    df['tokenized_text'] = df['combinedDescription'].apply(tokenize_text)
    df['combinedDescription'] = df['tokenized_text'].apply(tokens_to_str)

    return df



def main():

    opt = docopt(__doc__)
    raw_file_path = opt["--raw_path"]
    output_file_path = opt["--output_path"]
    exclude_sets = opt["--exclude_sets"]

    print("Loading and preprocessing raw data...")
    preprocessed_df = load_and_preprocess_raw_data(file_path=raw_file_path)
    if exclude_sets:
        print("Removing items containing sets...")
        preprocessed_df = exclude_set_items(preprocessed_df)
    print("Done preprocessing, saving results....")
    preprocessed_df.to_csv(output_file_path, index=False)
    print(f"Completed preprocessing successfully, data saved to: {output_file_path}")


if __name__ == "__main__":
    main()