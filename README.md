# Estimating Ground Truth in a Low-labelled Data Regime: A Study of Racism Detection in Spanish
[![DOI](https://zenodo.org/badge/487601308.svg)](https://zenodo.org/badge/latestdoi/487601308)

This repository contains the code for the paper submitted to 1st Workshop on Novel Evaluation Approaches for Text Classification Systems on Social Media ([NEATCLasS](https://neatclass-workshop.github.io/)).

Our paper investigates the impact of different ground truth estimation methods on a model for racism detection. The experiments are organised as follows: 
- The first notebook implements the ground truth estimates ([Ground Truth Estimation](./1_annotations_aggregation.ipynb)).
- The second notebooks assess the annotation problem due to the few labels and high disagreement ([Annotation Assessment](2_annotation_assessment.ipynb)).
- The last notebook evaluates the performance and error analysis of the models trained on different estimates  ([Impact Evaluation on Racism Detection](/3_racism_detection_model.ipynb)).

Most annotators may not have previous experience with racism, as only three belong to the Black community. Our empirical results show better performance at lower thresholds for classifying messages as racist, which may be due to the propagation of permissiveness in annotating racist messages to the model.

## Installation

The BcnAnalytics team is working for the release of the data. We will add the [link]() to the data repository here ⌛

Please copy the `evaluation_sample.csv` and `labels_racism.csv` to the data folder!

To run our code, pip install the packages in `requirements.txt`.

Then, you are ready to go for generating all the analysis outputs from our paper using the jupyter notebooks. We describe your resulting project directory tree below 👇

## File descriptions 

- **data**: Folder containing the data used in this work.
    - **predictions**: Folder with the predictions of all models in all epochs generated in Notebook 3.
    - **predictions_orig**: Folder with the predictions of best epoch of all models (_*evaluation_sample_m_vote_nonstrict.csv*_,_*evaluation_sample_raw.csv*_,_*evaluation_sample_w_m_vote_nonstrict.csv*_)
    - **toxicity_scores**: Evaluation sample with Perspective Toxicity scores and English translation perspective api notebook (_*evaluation_sample_translated.csv*_).
  - ***evaluation_sample.csv***: Evaluation data sample.
  - ***labels_racism.csv***: Raw data.
  - ***labels_racism_aggregated.csv***: Training data with the aggregated labels (m_vote and w_m_vote) from notebook 1.
  - ***labels_racism_preproc.csv***: Raw data with message ids and numeric labels from notebook 1.
  - ***ids_validation_set.json***: List of the id of the samples that belong to the validation set, used in Notebook 3.

- **model**: Folder with performance results of models at different tresholds for predicting racist labels.
  - ***thr_analysis_w_m_vote.csv***: F1 scores using different thresholds of the weighted majority vote.
  - ***thr_analysis_w_m_vote.png***: Plot of the F1 scores.
- **models**: Folder with trained models at each epoch.

- **src**: Folder with other functions and notebooks used in this work.
  - **_perspective_api.ipynb_**: Google API jupyter notebook for getting the toxicity of the messages.
  - **_huggingface.py_**: Python script with functions to load, train, and evaluate the models using HuggingFace transfomer library.
  - **_utils.py_**: Python script with utility functions for loading the dataset or binarizing the labels.
- **plots**: Folder with plots from data exploration in notebook 2.
  - _**agreement_annotators_nonstrict.png**_: Strict agreement plot.
  - _**agreement_annotators_strict.png**_: Non-strict agreement plot.
- _**1_annotations_aggregation.ipynb**_: First notebook to be executed. This notebook allows us to get the aggregated labels and save them in the data folder.
- _**2_annotation_assessment.ipynb**_: Second notebook to be executed. This notebook analyses the previous data for getting the agreement between annotators.
- _**3_racism_detection_model.ipynb**_: Third notebook to be executed. This last notebook analyses the threshold importance as well as the performance evaluation.

## Authors

Do not hesitate to contact us with any ideas or any reproducibility problems!

- [Paula Reyero Lobo](mailto:paula.reyero-lobo@open.ac.uk)
- [Martino Mensio](mailto:martino.mensio@open.ac.uk)
- [Angel Pavon Perez](mailto:angel.pavon-perez@open.ac.uk)
- [Vaclav Bayer](mailto:vaclav.bayer@open.ac.uk)
- [Joseph Kwarteng](mailto:joseph.kwarteng@open.ac.uk)

 
