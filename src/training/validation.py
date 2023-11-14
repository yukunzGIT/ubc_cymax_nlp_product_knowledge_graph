#!/usr/bin/env python
# authors: Lisa Sequeira, Althrun Sun, Luke Yang, Edward Zhang
# date: 2023-06-26

"""This script validates the model's performance using annother set of annotations

Usage: validation.py --anno_path=<anno_path> --model_path=<model_path>
Options:
--anno_path=<anno_path>         This is the path to the JSON annotated data 
--model_path=<model_path>       This is the path of the model to validate
"""
from docopt import docopt
import spacy
from seqeval.metrics import classification_report
from seqeval.scheme import BILOU
from spacy.training import offsets_to_biluo_tags
import json

def JSON_to_annotations(*files):
    """
    Converts JSON files containing annotations into a merged list of annotations.

    Parameters:
    *files (str): Variable number of file paths to JSON files.

    Returns:
    list: A merged list of annotations extracted from the provided JSON files.
    """
    merged_data = []
    for file in files:
        with open(file, 'r', encoding = 'utf-8') as f:
            load_data = json.load(f)
            merged_data += load_data["annotations"]

    return merged_data

def report(manual_annotations, model = "en_core_web_sm"):
    """
    Generates a classification report comparing manual annotations with predicted annotations.

    Parameters:
    manual_annotations (list): A list of text and annotation pairs.
    model (str): The name or path of the Spacy model to be used. Defaults to "en_core_web_sm".

    Returns:
    None
    """

    #manual_annotations = tagged_data
    nlp = spacy.load(model) #change to fine-tuned model
    true_BILOU_tags = []
    pred_BILOU_tags = []

    for text, annotation in manual_annotations:
        doc = nlp(text)
        #pulling out "offsets"
        pred_entities = [[ent.start_char, ent.end_char, ent.label_] for ent in doc.ents]

        if len(text) != 0:
            if type(pred_entities[0]) != list: #prevent an error when only a single entity
                pred_entities = [pred_entities]
                print(ent)

            pred_BILOU_tags.append(offsets_to_biluo_tags(doc, pred_entities))

            if type(annotation["entities"][0]) != list: #prevent an error when only a single entity
                ent = [annotation["entities"]]
            else:
                ent = annotation["entities"]
            true_BILOU_tags.append(offsets_to_biluo_tags(doc, ent))

    return print(classification_report(true_BILOU_tags,pred_BILOU_tags, mode='strict', scheme=BILOU))


def main():
    opt = docopt(__doc__)
    anno_path = opt["--anno_path"]
    model_path = opt["--model_path"]
    
    print("Loading validation data...")
    data_anno = JSON_to_annotations(anno_path)
    print("Generating classification report...")
    report(data_anno, model_path)


if __name__ == "__main__":
    main()