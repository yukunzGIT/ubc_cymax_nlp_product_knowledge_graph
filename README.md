Product Knowledge Graph
==============================

Authors:

- Lisa Sequeira
- Althrun Sun
- Luke Yang
- Yukun (Edward) Zhang

Introduction
------------

This repository serves as the data product delivered to Cymax Group Technologies in the 2023 Capstone project. In this project, we constructed a pipeline for training a large language model for named-entity recognition (NER), calculating similarities using [Annoy models](https://sds-aau.github.io/M3Port19/portfolio/ann/), and visualizing the results using Dash.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── annotations         <- Manual Annotations
    │   │    └──validation      <- Manual Annotations for validation
    │   ├── interim             <- Dataframe containing the NER from fine-tuned model (raw) 
    │       └── NER_post_processing <- Dataframe generated from post processing NER (remove duplicates etc.)
    │       └── raw_pre_processing  <- Pre-processing (ex: “combo” items removal) on raw dataset
    │       └── vect_emb_pipelines  <- Conversion of processed data to vector embedding from different pipelines
    │       └── extracted_entities  <- Dataframe containing the NER from fine-tuned model (raw)
    │   ├── config             <- Configurations for fine-tuning language models
    │   └── raw                <- The original, immutable data dump.
    │
    ├── dashboard          <- Dashboard program source code
    │── database           <- Neo4j graph database data and code
    ├── docs               <- Contains inter annotator guidelines and fine-tuned models notebook
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   ├── NER_model          <- NER models go here (from BERT/tok2vec)
        │   ├── pipeline            <- Best fine-tuned model
        │   └── static              <- Last fine-tuned model
    │   ├── similiarity_model  <- Models generated for the similarity search
    │       └── pipeline-models     <- Pipelines for transforming data
    │       └── annoy-models        <- ANNOY models for similarity search
    ├── notebooks          <- Jupyter notebooks (uncleaned). Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── outputs                  <- Generated graphics and figures to be used in reporting (inc. similarity df)
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment.
    │
    ├── setup.py           <- Make this project pip installable with `pip install -e`
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py               <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── pre_training       <- Scripts to turn raw data into features for model fine-tuning
    │   │   ├── json2spacy.py         <- Convert annotations to spacy format
    │   │   ├── IAA_validation.py     <- Inter annotator agreement score
    │   │   └── raw_pre_processing.py <- Processing to raw dataframe (combine title with descrip. etc.)
    │   │
    │   ├── post_training     <- Scripts to train models and then use trained models to extract entities               
    │   │   ├── extract_entities.py   <- generating the entities using fine-tuned model
    │   │   ├── NER_validation.py     <- EDA and classification report on generated entities
    │   │   └── post_processing.py    <- Remove duplicates, lemmatization
    │   │
    │   ├── similarity     <- Product-product similarity modeling and finding
    │   │   └── pipeline.py           <- pipelines for testing similarity
    │   │   └── ANNOY_model.py        <- similarity models that we will use
    │   │   └── sim_model_validation.py  <- generation of dataframe for similarity search evaluation
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py          <- NER, Similar search, and knowledge graph visualization (png image and csv files)
    └── test                <- Contains test scripts for functions in src. 

--------

## Usage

**Note: You may want to create a virtual environment first.**

We recommend running the project using Python 3.10 or higher. Before you execute the pipeline, you may need to set up the environment in `pip` by running the following command:  

```
pip install -r requirements.txt
```

### Example Usage

To demonstrate the functionality of our project's NER capability, located in the ```src``` folder, we provide the following example usage. Assuming that the current working directory is the root directory of this project, we can use the following code to initialize the model.

```
from src import EntityRecognizer
model = EntityRecognizer('models/NER_model/static/bert-uncased/')
```

We can predict a given text using the `model` we initialized. The prediction includes a list of tuples with start indices, end indices, and entity categories.

```
model.predict('This rectangular modern styled bed is black.')
>>> [{'start': 5, 'end': 16, 'label': 'APPEARANCE'},
 {'start': 17, 'end': 23, 'label': 'STYLE'},
 {'start': 31, 'end': 34, 'label': 'TYPE'},
 {'start': 38, 'end': 43, 'label': 'COLOR'}]
```

The above example prediction can also be visualized using Jupyter Notebook. We can run the following code in the Jupyter Notebook to visualize the named-entity recognition:

```
model.display('This rectangular modern styled bed is black.')
```

![ReadMe](reports/figures/readme_example.png)

## Running the Entire Analysis with New Annotated Data

This section introduces the training process of the model. If the user decides to add more annotated data to re-train the model, or to customize the entities different from the current seven we have, we implemented GNU make to facilitate the reproducibility of this project. The pipeline of NER model training, entity extraction and cleaning could be run using make.

### Run the Analysis with Make

If your machine has CUDA available, the pipeline will train a transformer-based model (BERT). If you are using a CPU-only machine, the pipeline will train a tok2vec model instead. The following command will execute the sequence of analysis.

```
make
```

Running the following command from the
root directory of this project could clean up the analysis to its initial state:

```
make clean
```

### Note for Setting up CUDA

This section is for debugging if you have a GPU machine but cannot train the transformer model. You may need to install the proper CUDA driver when running the pipeline. If you have GPU on your machine but the pipeline still executes with tok2vec model, you may need to install the CUDA ToolKit [here](https://developer.nvidia.com/cuda-11-8-0-download-archive) and make sure you have proper C++ compilers installed using [Microsoft Visual Studio Tools](https://visualstudio.microsoft.com/downloads/)

## Running Partial Analysis

### Fine-tuning Other Models

Currently, we have fine-tuned 4 models based on CUDA in Google Colabs. Since the model results would be slightly different each time we re-run the fine-tuning process, we only save our best-performing model for our final pipeline. If you would like to train other models by yourself, we also provide you with the following options:

#### Option 1

Only if you have CUDA installed on your machine, you can run the following command to re-train the other 3 under-performing models by yourself locally. The training time normally takes a while and the fine-tuned models would be saved in the repo file path ```models/NER_model/pipeline```. Since these 3 saved models are over 1 GB each, we recommend having at least 5 GB storage available for your machine.

If you would like to re-train the bert-large-uncased model based on our 200 samples, you can run this command:

```
make train_large_bert
```

If you would like to re-train the Jean-Baptiste/roberta-large-ner-english model based on our 200 samples, you can run this command:

```
make training_roberta
```

If you would like to re-train the dslim/bert-base-NER model based on our 200 samples, you can run this command:

```
make training_bert_ner
```

#### Option 2

If you do not have CUDA installed on your machine, we recommend you set up Google Colabs and push the repo up. Then you can adapt and run this [notebook](docs/fine-tune_additional_models.ipynb) to re-train the above 3 models.

### Extract Entities

If you have a `.csv` file that contains a list of products with their corresponding titles and descriptions, this section explains how to use the existing fine-tuned model to extract the entities. First, you can process the raw data using:

```
python src/pre_training/raw_pre_processing.py --raw_path=<raw_path> --output_path=<output_path>
```

Then, you can get the `.csv` with the extracted entities using

```
python src/post_training/extract_entities.py --base_dataframe=<base_dataframe> --model_path=models/NER_model/static/bert-uncased --output_path=<output_path>
```

If you wish to further clean the extracted entities, you can use

```
python src/post_training/post_processing.py --base_dataframe=<base_dataframe> --output_path==<output_path>
```

--------

## Dashboard

To better show the structure and components of our product knowledge map, we built a dashboard using python dash and networkX. This dashboard mainly displays two important relationships: product-product, and product-entities. via the search bar on the top right You can find a specific product by its productID and show the top 3, 5, 8 similar products. The gif shown below is the introduction of usage for the product knowledge graph dashboard.

![product knowledge graph dashboard](dashboard/img/dash_demo.gif)

### Running the Dashboard Locally

Before you want to locally run the dashboard, you want to make sure that you installed all the requirements listed in the previous section (e.g. running the following command).
Once all requirements have been installed, you can initialize the dashboard by running

```
python dashboard/src/app.py 
```

and follow the link provided to access the dashboard.

--------

## High-performance Graph Database

Efficient retrieval of the knowledge graph is an important challenge, and how to build an efficient knowledge graph storage architecture is the key to improving the stability and accuracy of the recommendation system. We adopt the Neo4j graph database and combine it with Cypher query language to build a high-performance commodity knowledge graph database management system. The gif shown below is a demo of the Neo4j database.

![product knowledge graph dashboard](dashboard/img/neo4j_demo.gif)


## License

Distributed under the MIT License. See `LICENSE` for more information.
