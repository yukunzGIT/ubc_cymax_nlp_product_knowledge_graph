import os
import sys
import pytest
# cur_dir = os.getcwd()
# src_path = cur_dir[
#     : cur_dir.index("ubc_capstone_productknowledgegraph") + len("ubc_capstone_productknowledgegraph")
# ]
# if src_path not in sys.path:
#     sys.path.append(src_path)
from src.pre_training.raw_pre_processing import *

def test_tokens_to_str():
    assert tokens_to_str(['Hello','World']) == 'Hello World'
    assert tokens_to_str(['1','2','3']) == '1 2 3'
    