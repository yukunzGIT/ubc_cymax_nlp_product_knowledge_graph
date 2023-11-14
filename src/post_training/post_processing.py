#!/usr/bin/env python
# authors: Lisa Sequeira, Althrun Sun, Luke Yang, Edward Zhang
# date: 2023-06-12

"""This script help clean up extracted entities
Usage: post_processing.py --base_dataframe=<base_dataframe> --output_path=<output_path> 
Options:
--base_dataframe=<base_dataframe> This is dataframe you want to extract your entities on
--output_path=<output_path>       This is the path of result csv
"""
from docopt import docopt
import re
import spacy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ENTITIES = [
    "COLOR",
    "TYPE",
    "STYLE",
    "APPEARANCE",
    "ADDITIONAL_MATERIAL",
    "FEATURE",
    "NOTICE",
]
CAT_COLUMNS = ["PrimaryMaterial", "DescAccCategoryLevel2", "DescCategoryLevel5"]
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def replace_non_alphanumeric(string):
    """
    Removes characters that are not English letters or numbers from a string.

    Parameters:
    string (str): The input string to process.

    Returns:
    str: The processed string with non-alphanumeric characters removed.
    """
    pattern = "[^a-z0-9,'' ]"
    return re.sub(pattern, "", string)


def lemmatize(string):
    """
    Lemmatizes a given string using the English language model in spaCy.

    Parameters:
    string (str): The input string to lemmatize.

    Returns:
    str: The lemmatized string.
    """
    text = " ".join([token.lemma_ for token in nlp(string)])
    return text


def process_entity_columns(df, *col_names, sent_intro=True):
    """
    Processes the entity columns in a pandas DataFrame by converting strings to lowercase, removing punctuation,
    replacing non-alphanumeric characters, lemmatizing, and joining entities.

    Parameters:
    df (pandas.DataFrame): The DataFrame to process.
    col_names (str): Variable-length arguments representing the names of the entity columns.
    sent_intro (bool): Indicates whether to add an introduction to the sentence.

    Returns:
    pandas.DataFrame: The processed DataFrame.
    """

    col_count = len(col_names)
    i = 1
    for col_name in col_names:
        # print out progress
        print("Progress: (" + str(i) + "/" + str(col_count) + ").")
        i += 1
        # Convert all strings in each row to lowercase and remove periods, semicolons, and dashes
        col_data = (
            df[col_name]
            .str.lower()
            .str.replace(r"[.:;-]", " ", regex=True)
            .str.replace("'", "")
        )

        # Remove characters that are not English letters or numbers
        col_data = col_data.apply(lambda x: replace_non_alphanumeric(x))

        # Lemmatize all the entities
        col_data = col_data.apply(lambda x: lemmatize(x))

        col_data = col_data.apply(lambda x: ", ".join(set(x.split(" , "))))

        df[col_name] = col_data

    return df


def sentence_transform(df, *col_names, sent_intro=True):
    """
    Transforms the entity columns in a pandas DataFrame by removing duplicates, joining entities with 'and',
    and optionally adding an introduction to the sentence.

    Parameters:
    df (pandas.DataFrame): The DataFrame to transform.
    col_names (str): Variable-length arguments representing the names of the entity columns.
    sent_intro (bool): Indicates whether to add an introduction to the sentence.

    Returns:
    pandas.DataFrame: The transformed DataFrame.
    """

    for col_name in col_names:
        # Remove duplicate entities and create a sentence to join all entities with 'and'
        col_data = df[col_name].apply(lambda x: " and ".join(set(str(x).split(", "))))

        # Only add an introduction to the sentence if requested
        if sent_intro:
            col_data = col_data.apply(
                lambda x: f"The {col_name.lower().replace('_', ' ')} of this product is {str(x)}"
                if x
                else ""
            )

        df[col_name + "_sentence"] = col_data

    return df


def process_categorical_columns(df, *col_names):
    """
    Processes the categorical columns in a pandas DataFrame by converting strings to lowercase,
    and replacing specific delimiters with commas.

    Parameters:
    df (pandas.DataFrame): The DataFrame to process.
    col_names (str): Variable-length arguments representing the names of the categorical columns.

    Returns:
    pandas.DataFrame: The processed DataFrame.
    """

    col_count = len(col_names)
    for col_name in col_names:
        # Convert all strings in each row to lowercase and replace specific delimiters with commas
        df[col_name] = (
            df[col_name].str.lower().str.replace(" / ", ", ").str.replace(" & ", ", ")
        )

    return df


def remove_exact_match(df, col_name, col_name_subset):
    """
    Removes exact matches from a column in a pandas DataFrame based on another column's values.

    Parameters:
    df (pandas.DataFrame): The DataFrame to modify.
    col_name (str): The name of the column from which to remove exact matches.
    col_name_subset (str): The name of the column used to determine the exact matches to remove.

    Returns:
    pandas.DataFrame: The modified DataFrame.
    """
    col_name_list_series = df[col_name].apply(lambda x: str(x).split(", "))
    col_name_subset_list_series = df[col_name_subset].apply(
        lambda x: str(x).split(", ")
    )

    for i in range(len(df[col_name])):
        col_name_list_series[i] = [
            item
            for item in col_name_list_series[i]
            if item not in col_name_subset_list_series[i]
        ]

    df[col_name] = col_name_list_series.apply(lambda x: ", ".join(x))

    return df


def remove_brackets(df, text_columns):
    """
    Removes brackets from the beginning and end of text columns in a pandas DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to modify.
    text_columns (list): A list of column names containing text.

    Returns:
    pandas.DataFrame: The modified DataFrame.
    """
    for column in text_columns:
        df[column] = df[column].str.strip("[]")
    return df


def get_null_eda(df, fig_path="reports/figures/null_eda.png"):
    """
    Performs exploratory data analysis on null values in the NER (Named Entity Recognition) columns of a DataFrame.
    Generates a horizontal bar plot showing the null percentage in each NER column.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the NER columns.
    fig_path (str): Optional. The file path to save the generated plot. Defaults to 'reports/figures/null_eda.png'.

    Returns:
    None
    """
    # Calculate the number of null rows in each NER column
    null_rows = df[ENTITIES].apply(lambda x: x.str.len() == 0).sum()

    # Create a DataFrame to store the NER column names and their corresponding null counts
    null_df = pd.DataFrame(
        {"NER Tag Columns": null_rows.index, "Null Counts": null_rows.values}
    )

    # Calculate the null percentage for each NER column
    null_df["Null Percentage"] = round(null_df["Null Counts"] / len(df) * 100, 2)

    # Generate a colormap for the plot
    cmap = plt.colormaps.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, len(null_df)))

    # Create a horizontal bar plot to visualize the null percentage in each NER column
    null_df.sort_values(by="Null Percentage", ascending=True).plot.barh(
        x="NER Tag Columns",
        y="Null Percentage",
        rot=0,
        title="Null Percentage in NER Columns",
        ylabel="NER Columns",
        xlabel="Null Percentage of all rows (%)",
        legend=False,
        color=colors,
    )

    # Save the generated plot to the specified file path
    plt.savefig(fig_path)


def main():
    opt = docopt(__doc__)
    base_dataframe = opt["--base_dataframe"]
    output_path = opt["--output_path"]

    print("Loading data frame...")
    df = pd.read_csv(base_dataframe)
    print("Removing extra brackets...")
    df = remove_brackets(df, ENTITIES)
    get_null_eda(df)
    print("Processing entity columns, note this may take awhile...")
    df = process_entity_columns(df, *ENTITIES)
    print("Processing categorical columns...")
    df = process_categorical_columns(df, *CAT_COLUMNS)
    print("Romoving exact match...")
    df = remove_exact_match(df, "ADDITIONAL_MATERIAL", "PrimaryMaterial")
    df = sentence_transform(
        df,
        "COLOR",
        "TYPE",
        "STYLE",
        "APPEARANCE",
        "ADDITIONAL_MATERIAL",
        "FEATURE",
        "PrimaryMaterial",
    )
    df = sentence_transform(df, "NOTICE", sent_intro=False)
    df.to_csv(output_path)


if __name__ == "__main__":
    main()
