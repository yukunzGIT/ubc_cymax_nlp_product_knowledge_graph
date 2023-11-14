import os
import sys
import pytest

from src.similarity.pipeline import *

def test_pipeline_output_type(): 
    sent_features = ["Test_column"]
    p1 = build_pipelines(sent_features)

    assert type(p1) ==  ColumnTransformer

def test_pipeline_output_num_bin(): 
    numeric_found = False
    binary_found = False

    sent_feat = ["Test_column"]
    bin_feat = ["Test Bin column"]
    num_feat = ["Test Num column"]

    p2 = build_pipelines(sent_feat, bin_feat, num_feat )

    for name,_,_ in p2.transformers:
        if 'num' in name:
            numeric_found = True
        elif 'binary' in name:
            binary_found = True

    assert numeric_found, "Numeric component not added in ColumnTransformer"
    assert binary_found, "Binary component not added in ColumnTransformer"