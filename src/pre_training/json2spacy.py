#!/usr/bin/env python
# authors: Lisa Sequeira, Althrun Sun, Luke Yang, Edward Zhang
# date: 2023-06-07

"""This script loads JSON annotations and split the data and exports to spacy format.

Usage: json2spacy.py --anno_path=<anno_path> --output_dir=<output_dir> 
Options:
--anno_path=<anno_path>         This is the path to the JSON annotated data
--output_dir=<output_dir>       This is the directory of train/test split
"""
from docopt import docopt
from typing import Tuple, List, Dict
import spacy
from spacy.tokens import DocBin
import pandas as pd
import json
from sklearn.model_selection import train_test_split

nlp = spacy.blank("en")

def merge_json_files(*files: str) -> List[Dict]:
    """
    Merge multiple JSON files containing annotation data.

    Parameters:
        *files (str): Variable number of file paths to JSON files.

    Returns:
        List[Dict]: A list of merged annotation data.

    Note:
        - Each JSON file should have a key "annotations" containing the annotation data.
        - Empty labels will be removed from the merged data.
    """
    merged_data = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            load_data = json.load(f)
            merged_data += load_data["annotations"]

    # Remove empty labeling
    res = []
    for d in merged_data:
        if d[-1]['entities'] != []:
            res.append(d)
    return res


def json_to_spacy(data: List[Tuple[str, Dict]], file_name: str) -> None:
    """
    Convert JSON annotation data to the spaCy format and save it as a binary file.

    Parameters:
        data (List[Tuple[str, Dict]]): List of tuples containing text and annotation data.
        file_name (str): Name of the output file to save the converted data.

    Returns:
        None

    Note:
        - The annotation data should be in the format {"entities": [[start, end, label], ...]}.
        - The converted data will be saved as a spaCy binary file.
    """
    db = DocBin()
    
    for text, annotations in data:
        doc = nlp(text)
        ents = []
        for start, end, label in annotations["entities"]:
            span = doc.char_span(start, end, label=label)
            ents.append(span)
        if None in ents:
            for e in ents:
                print(e)
        doc.ents = ents
        db.add(doc)
    db.to_disk(file_name + ".spacy")



def main():
    opt = docopt(__doc__)
    anno_path = opt["--anno_path"]
    output_dir = opt["--output_dir"]

    print("Loading JSON annotations...")
    data = merge_json_files(anno_path)
    print("Transforming to spacy format...")
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=123)
    json_to_spacy(train_data,output_dir+"train")
    json_to_spacy(test_data,output_dir+"test")


if __name__ == "__main__":
    main()