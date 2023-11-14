import subprocess
import sys
import torch

if torch.has_cuda:
    print('We notice that the system has CUDA available. We will use BERT as our base model to fine-tune.')
    subprocess.check_call([sys.executable, '-m', 'spacy', 'train', 'data/config/config_spacy_gpu.cfg', '--paths.train', "data/annotations/spacy/v_final/train.spacy", '--paths.dev', "data/annotations/spacy/v_final/test.spacy", '--gpu-id', '0', '--output', "models/NER_model/pipeline"])
else:
    print('We notice that the system does not has CUDA available. We will use tok2vec as our base model to fine-tune.')
    subprocess.check_call([sys.executable, '-m', 'spacy', 'train', 'data/config/config_spacy_cpu.cfg', '--paths.train', "data/annotations/spacy/v_final/train.spacy", '--paths.dev', "data/annotations/spacy/v_final/test.spacy", '--output', "models/NER_model/pipeline"])