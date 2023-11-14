#!/usr/bin/env python
# authors: Lisa Sequeira, Althrun Sun, Luke Yang, Edward Zhang
# date: 2023-06-07

"""This script help extract entities using a trained model and a processed dataframe
Usage: extract_entities.py --base_dataframe=<base_dataframe> --model_path=<model_path> --output_path=<output_path> 
Options:
--base_dataframe=<base_dataframe> This is dataframe you want to extract your entities on
--model_path=<model_path>         This is the path to trained model
--output_path=<output_path>       This is the path of result csv
"""
from docopt import docopt
from tqdm import tqdm
import spacy
import os
import pandas as pd
import json

def extract_entities(df: pd.DataFrame, model: spacy.language.Language):
    """
    Extracts entities from the given DataFrame using a specified Spacy model.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to process.
        model (spacy.language.Language): The Spacy model used to process the text and extract entities.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted entities.
    """
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    with open('data/annotations/200_samples/annotations.json', 'r', encoding='utf-8') as f:
        tag_list = json.load(f)['classes']
    # Create a function to process text and extract entities
    def process_text(idProduct, text):
        entities = {
            "idProduct": idProduct,
        }
        
        for tag_name in tag_list:
            entities[tag_name]=[]
        
        doc = model(text)
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
        return entities

    # Process text and extract entities
    entities_list = []
    for idProduct, text in tqdm(zip(df['idProduct'], df['combinedDescription'])):
        entities = process_text(idProduct, text)
        entities_list.append(entities)

    # Convert entities to DataFrame
    df_entities = pd.DataFrame(entities_list)
    return df_entities


def main():
    opt = docopt(__doc__)
    base_dataframe = opt["--base_dataframe"]
    model_path = opt["--model_path"]
    output_path = opt["--output_path"]

    print("Loading data frame...")
    df = pd.read_csv(base_dataframe)
    print("Loading model...")
    model = spacy.load(model_path)
    print("Extracting entities...")
    print("In total "+str(len(df))+' items to extract...')
    df_entities_final = extract_entities(df,model)
    extract_ent_df= pd.concat([df, df_entities_final], axis=1)
    extract_ent_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()