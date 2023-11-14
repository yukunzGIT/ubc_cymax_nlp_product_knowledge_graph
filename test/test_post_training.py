import os
import sys
import pytest

from src.post_training.post_processing import *

def test_process_entity_columns(): 
    x = "'ENTITY','entity', 'Other'"
    y = 'entity, other'
    test_df = pd.DataFrame({'input_col':[x], 'output_col':[y]})
    test_df = process_entity_columns(test_df, 'input_col', sent_intro = False)
    assert test_df['input_col'].equals(test_df['output_col'])
