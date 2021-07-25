ISS610: Applied Machine Learning Project
==============================

Chest x-ray automated diagnosis machine learning project.

Getting Started
------------

Create the environment from the environment.yml file:

`conda env create -f chexpert.yml`

Usage
------------
To run the ml jobs, simply run the following

```
usage: /bin/run_ml_chexpert.py [-h] [--pca {True,False}] [--pca_pretrained PCA_PRETRAINED] [--pca_n_components PCA_N_COMPONENTS] [--preprocessing PREPROCESSING] [--file FILE] [--map {U-zero,U-one,Random}] [--batchsize BATCHSIZE] [--validsize VALIDSIZE] [--limit LIMIT] [--path PATH]
                          [--ylabels YLABELS [YLABELS ...]] [--model {MultinomialNB,SGDClassifier}] [--model_pretrained MODEL_PRETRAINED] [--n_jobs N_JOBS]

optional arguments:
  -h, --help            show this help message and exit
  --pca {True,False}    Training for pca
  --pca_pretrained PCA_PRETRAINED
                        .sav file path for pretrained pca model
  --pca_n_components PCA_N_COMPONENTS
                        n_components for pca
  --preprocessing PREPROCESSING
                        File path for image preprocessing
  --file FILE           Filename prefix
  --map {U-zero,U-one,Random}
                        Option for mapping uncertain labels
  --batchsize BATCHSIZE
                        Training batch size
  --validsize VALIDSIZE
                        Validation dataset size
  --limit LIMIT         Maximum dataset size capped
  --path PATH           Base path
  --ylabels YLABELS [YLABELS ...]
                        Labels to predict
  --model {MultinomialNB,SGDClassifier}
                        Choice of model
  --model_pretrained MODEL_PRETRAINED
                        .sav file for pretrained classifer e.g NaiveBayes_50_50_Random_1259_25072021.sav
  --n_jobs N_JOBS       Number of cores for multi-processing

```
###  Examples 

* To train a new PCA with pca components of 30 and Naive Bayes Multinomial Model with 'U-zero' option, with batch size of 50 and 10000 training dataset, 
for labels *"Edema", "No Finding""*, with prefix filename of `pca_nb`, and with image preprocessing config of 
`crop_median_blur_normalize_rotate_zoom.yaml`

    `python run_ml_chexpert.py 
--batchsize 50 
--limit 10000 
--model MultinomialNB 
--pca True --pca_n_components 30 
--preprocessing "config/crop_median_blur_normalize_rotate_zoom.yaml" 
--ylabels "No Finding" "Edema" 
--map U-zero 
--file pca_nb`

* To load a pretrained PCA and NaiveBayes model for labels *"Edema", "No Finding""*, with prefix filename of `pca_nb`, and with image preprocessing config of 
`crop_median_blur_normalize_rotate_zoom.yaml`. As the image preprocessing pipeline and y labels 
are not part of the model, you will need to remember which image preprocessing pipeline and choices of y labels 
you used earlier during training.

`python run_ml_chexpert.py 
--batchsize 50 
--limit 10000 
--model MultinomialNB 
--pca True 
--pca_pretrained "IncrementalPCA_50_100_Random_1431_25072021.sav" 
--model_pretrained "NaiveBayes_50_50_Random_1259_25072021.sav" 
--preprocessing "config/crop_median_blur_normalize_rotate_zoom.yaml" 
--ylabels "No Finding" "Edema" 
--file pca_nb`

Configurations
------------
The list of `.yaml` image preprocessing pipeline is under `/config/`

```
transformations:
  - [crop, {size: [320,320]}]
  - [median_blur,{}]
  - [normalize, {}]
  - [rotate, {}]
  - [zoom, {}]
  - [flatten,{}]
```

### Models

Additional sklearn models can be added under `/src/models/sklearn_models.py`
 

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── chexpert.yml   <- The conda environment file required
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Dataset classes for data loading and preprocessing
    │   │   └── dataset.py
    │   │
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── sklearn_models.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
