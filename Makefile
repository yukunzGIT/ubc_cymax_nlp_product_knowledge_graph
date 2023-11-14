# Product Knowledge Graph
# author: Lisa Sequeira, Althrun Sun, Luke Yang, Edward Zhang
# date: 2023-06-01

.PHONY: clean data lint setup

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = UBC_CAPSTONE_PRODUCTKNOWLEDGEGRAPH
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

all: setup sim_df

## Install Python Dependencies
setup: 
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m spacy download en_core_web_sm

sim_df: annoy_model
	$(PYTHON_INTERPRETER) src/similarity/sim_model_validation.py --original_df_path=data/interim/NER_post_processing/process_ent_df.csv --validation_df_path=data/raw/ProductDetails_UBC_GroupingData.csv --array_path=data/interim/vect_emb_pipelines/ --model_path=models/similarity_model/annoy-models/ --output_path=data/output/similarity.csv

annoy_model: pipeline
	$(PYTHON_INTERPRETER) src/similarity/annoy_model.py --array_path=data/interim/vect_emb_pipelines/ --output_model_path=models/similarity_model/annoy-models/

pipeline: data/interim/NER_post_processing/process_ent_df.csv
	$(PYTHON_INTERPRETER) src/similarity/pipeline.py --base_dataframe=data/interim/NER_post_processing/process_ent_df.csv

data/interim/NER_post_processing/process_ent_df.csv: data/interim/extract_entities/extract_ent_df.csv
	$(PYTHON_INTERPRETER) src/post_training/post_processing.py --base_dataframe=data/interim/extracted_entities/extract_ent_df.csv --output_path=data/interim/NER_post_processing/process_ent_df.csv

data/interim/extract_entities/extract_ent_df.csv: train validate
	$(PYTHON_INTERPRETER) src/post_training/extract_entities.py --base_dataframe=data/interim/raw_pre_processing/single_items.csv --model_path=models/NER_model/pipeline/model-best --output_path=data/interim/extracted_entities/extract_ent_df.csv

train: process_data data/annotations/spacy/v_final/train.spacy data/annotations/spacy/v_final/test.spacy iaa
	$(PYTHON_INTERPRETER) src/training/training.py

validate: train
	$(PYTHON_INTERPRETER) src/training/validation.py --anno_path=data/annotations/validation/validation.json --model_path=models/NER_model/pipeline/model-best

# train other 3 models only if the user has the need. 
train_large_bert: process_data data/annotations/spacy/v_final/train.spacy data/annotations/spacy/v_final/test.spacy
	$(PYTHON_INTERPRETER) src/training/training_large_bert.py

training_bert_ner: process_data data/annotations/spacy/v_final/train.spacy data/annotations/spacy/v_final/test.spacy
	$(PYTHON_INTERPRETER) src/training/training_bert_ner.py

training_roberta: process_data data/annotations/spacy/v_final/train.spacy data/annotations/spacy/v_final/test.spacy
	$(PYTHON_INTERPRETER) src/training/training_roberta.py

process_data: data/interim/raw_pre_processing/single_items.csv data/interim/raw_pre_processing/processed_data.csv

iaa: data/annotations/annotation_agreement_test/iaa_sample_1.json data/annotations/annotation_agreement_test/iaa_sample_2.json
	$(PYTHON_INTERPRETER) src/pre_training/inter_annotator_agreement.py --anno_path_1=data/annotations/annotation_agreement_test/iaa_sample_1.json --anno_path_2=data/annotations/annotation_agreement_test/iaa_sample_2.json

# pre-process data with only single items (e.g., scale and split into train & test)
data/interim/raw_pre_processing/single_items.csv : data/raw/ProductDetails_UBC.csv
	$(PYTHON_INTERPRETER) src/pre_training/raw_pre_processing.py --raw_path="data/raw/ProductDetails_UBC.csv" --output_path="data/interim/raw_pre_processing/single_items.csv" --exclude_set=True

# pre-process data with all items (e.g., scale and split into train & test)
data/interim/raw_pre_processing/processed_data.csv : src/pre_training/raw_pre_processing.py data/raw/ProductDetails_UBC.csv
	$(PYTHON_INTERPRETER) src/pre_training/raw_pre_processing.py --raw_path="data/raw/ProductDetails_UBC.csv" --output_path="data/interim/raw_pre_processing/processed_data.csv"

# train model using annotations
data/annotations/spacy/v_final/train.spacy data/annotations/spacy/v_final/test.spacy: data/annotations/200_samples/annotations.json
	$(PYTHON_INTERPRETER) src/pre_training/json2spacy.py --anno_path="data/annotations/200_samples/annotations.json" --output_dir='data/annotations/spacy/v_final/'

clean: 
	rm -f data/interim/raw_pre_processing/*.csv
	rm -f data/annotations/spacy/v_final/*.spacy
	rm -r models/NER_model/pipeline/model*
	rm -f reports/assets/*
