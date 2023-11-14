import subprocess
import sys
import torch

if torch.has_cuda:
    print('We notice that the system has CUDA available. Ready to proceed.')
    subprocess.check_call([sys.executable, '-m', 'spacy', 'train', 'data/config/archive_config_files/config_spacy_rbta_transf.cfg', '--paths.train', "data/annotations/spacy/v_final/train.spacy", '--paths.dev', "data/annotations/spacy/v_final/test.spacy", '--gpu-id', '0', '--output', "models/NER_model"])
else:
    print('We notice that the system does not has CUDA available. Please have CUDA installed.')
    