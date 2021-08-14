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
usage: run_ml_chexpert.py [-h] [--pca {True,False}] [--pca_pretrained PCA_PRETRAINED] [--pca_n_components PCA_N_COMPONENTS] [--preprocessing PREPROCESSING] [--file FILE] [--map {U-zero,U-one,Random}] [--batchsize BATCHSIZE] [--validsize VALIDSIZE] [--limit LIMIT] [--path PATH]
                          [--ylabels {No Finding,Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,Lung Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture,Support Devices} [{No Finding,Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,Lung Lesion,Edema,
Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture,Support Devices} ...]]
                          [--cnn_transfer {0,1}] [--frontal_only {0,1}] [--cnn {True,False}] [--cnn_pretrained CNN_PRETRAINED] [--cnn_model {ResNet152_new,DenseNet121_new,MobileNetv2_keras,MobileNetv2_pop1,DenseNet121_keras,ResNet152_keras}] [--cnn_param CNN_PARAM]
                          [--model {MultinomialNB,GaussianNB,RandomForestClassifier,SGDClassifier,SGDClassifier_Elastic}] [--model_pretrained MODEL_PRETRAINED] [--n_jobs N_JOBS] [--epochs EPOCHS] [--steps_execute STEPS_EXECUTE] [--layer_train LAYER_TRAIN]

optional arguments:
  -h, --help            show this help message and exit
  --pca {True,False}    Option to train pca model.
  --pca_pretrained PCA_PRETRAINED
                        .sav file path for pretrained pca model.
  --pca_n_components PCA_N_COMPONENTS
                        n_components for pca.
  --preprocessing PREPROCESSING
                        File path for image preprocessing.
  --file FILE           Filename prefix. You should give a meaningful name for easy tracking.
  --map {U-zero,U-one,Random}
                        Option for mapping uncertain labels.
  --batchsize BATCHSIZE
                        Training batch size.
  --validsize VALIDSIZE
                        Validation dataset size.
  --limit LIMIT         Maximum dataset size capped.
  --path PATH           Base path.
  --ylabels {No Finding,Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,Lung Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture,Support Devices} [{No Finding,Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,Lung Lesion,Edema,Consolidation,Pneumonia,A
telectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture,Support Devices} ...]
                        Labels to predict.
  --cnn_transfer {0,1}  1 to have transfer learning, 0 to train from scratch
  --frontal_only {0,1}  1 to have frontal view only, 0 to include both frontal/lateral views
  --cnn {True,False}    'True' if running CNN model.
  --cnn_pretrained CNN_PRETRAINED
                        model file path for pretrained cnn model.
  --cnn_model {ResNet152_new,DenseNet121_new,MobileNetv2_keras,MobileNetv2_pop1,DenseNet121_keras,ResNet152_keras}
                        Choice of cnn model.
  --cnn_param CNN_PARAM
                        .yaml config file for model hyperparameter
  --model {MultinomialNB,GaussianNB,RandomForestClassifier,SGDClassifier,SGDClassifier_Elastic}
                        Choice of model.
  --model_pretrained MODEL_PRETRAINED
                        .sav file for pretrained classifer e.g NaiveBayes_50_50_Random_1259_25072021.sav
  --n_jobs N_JOBS       Number of cores for multi-processing.
  --epochs EPOCHS       Number of epochs.
  --steps_execute STEPS_EXECUTE
                        Number of steps per execution.
  --layer_train LAYER_TRAIN
                        Number of layer groups to train for each step during 1st epoch.



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

* To train a CNN model (MobileNetv2) with batch size of 16, limit of 5000 images in training dataset, without transfer learning, default hyperparameter 
and epochs = 5, and for tensorflow to run evaluation metric with step of 1 iteration, to do unfreeze weights for 20 groups of layers,
using label mapping of U-one 

`python run_ml_chexpert.py --batchsize 16 --epochs 5 --steps_execute 1 --layer_train 20 --cnn_model DenseNet121_keras --cnn_transfer 0 --map U-one --cnn True --file cnn_standard_balanced_fulldata_gradual20 --preprocessing cnn_standard.yaml`


Configurations
------------

### Image Preprocessing 
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

### CNN hyperparameter 
The list of `.yaml` cnn model hyperparameter config is under `/config/`

```
optimizer: Adam
learning_rate: 0.0001
loss: binary_crossentropy
```

### Models

* Additional sklearn models can be added under `/src/models/sklearn_models.py`
 * Additional Tensorflow models can be added under `/src/models/tensorflow_models.py`


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
    │   │   └── batchloader.py
    │   │   └── imgproc.py
    │   │   └── imgproc_skimage.py    
    │   │
    │   │
    │   ├── models         <- Scripts to setup model configurations
    │   │   ├── sklearn_models.py
    │   │   ├── tensorflow_models.py
    │   │
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
