#!/usr/bin/env python
# authors: Lisa Sequeira, Althrun Sun, Luke Yang, Edward Zhang
# date: 2023-06-26

"""This script checks for inter-annotator agreements

Usage: inter_annotator_agreement.py --anno_path_1=<anno_path_1> --anno_path_2=<anno_path_2>   
Options:
--anno_path_1=<anno_path_1>         This is the path to the first JSON annotated data 
--anno_path_2=<anno_path_2>         This is the path to the secondJSON annotated data 
"""
from docopt import docopt
import numpy as np
import json
from sklearn.metrics import cohen_kappa_score

def multi_cohen_kappa_score(annotations1, annotations2):

    pair_scores = []
    scores_per_item = []
    for index in range(len(annotations1["annotations"])):
        
        for pairs1 in annotations1["annotations"][index][1]["entities"]:
            for pairs2 in annotations2["annotations"][index][1]["entities"]:
                # compare the cohen_kappa_score of the annoation pairs with the same starting index
                if pairs1[0] == pairs2[0]:
                    subscore = cohen_kappa_score(pairs1, pairs2)
                    break
                else:
                    # if annoation pairs do not have same starting index, then this pair cohen_kappa_score==0
                    subscore = 0
            pair_scores.append(subscore)

        scores_per_item.append(np.mean(pair_scores))

    total_score = np.mean(scores_per_item)
    return total_score

def main():
    opt = docopt(__doc__)
    anno_path_1 = opt["--anno_path_1"]
    anno_path_2 = opt["--anno_path_2"]

    print("Performing inter-annotator agreement (IAA) check...")

    with open(anno_path_1, encoding='utf-8') as file:
        annotations1 = json.load(file)

    with open(anno_path_2, encoding='utf-8') as file:
        annotations2 = json.load(file)

    print("Calculating IAA score...")
    score = multi_cohen_kappa_score(annotations1, annotations2)

    print('The IAA score is calculated to be', round(score,3))


if __name__ == "__main__":
    main()