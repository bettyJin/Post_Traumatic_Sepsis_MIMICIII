# Rare Event Early Prediction: Sepsis Onset for Critically Ill Trauma Patients

### Overview
This repository contains the code for generating a dataset to predict the early onset of hospital-acquired sepsis among critically ill patients within a trauma cohort from the MIMIC III dataset.

The implementation is purely in Python, based on the methodology described in [TODO: link] paper. We utilize the Google BigQuery server to access and manage the MIMIC III database.

### Dataset
We introduce a publicly available, arguably reliable dataset derived from the MIMIC-III database. This dataset encompasses a well-defined trauma cohort with corresponding sepsis labels with concrete time stamps, specifically targeting the task of early hospital-acquired sepsis onset prediction among critically ill patients with trauma. We offer comprehensive data analysis and medical context for both the cohort and the sepsis labels, along with well-formed time-series vital sign features according to a deployable setup for ICUs.

---

# Usage

The project includes several scripts to perform different tasks:

### Check MIMIC Dataset Access
Verify access to the MIMIC III dataset through Google BigQuery:
```
python tests/test_mimic_access.py 
```

### Cohort Extraction: Critically Ill Trauma Patients
Extract the Critically Ill Trauma Patients cohort from the MIMIC III dataset:
```
python scripts/cohort_extraction.py 

```

### Assign Post-trauma Sepsis Labels
Assign Post-trauma Sepsis labels to the trauma cohort based on the defined criteria:
```
python scripts/assign_sepsis_label.py

```


### Generate Dataset
Generate a dataset according to the Early Sepsis Onset Prediction Setup:
```
python scripts/generate_dataset.py

```

### Benchmarks
Run benchmark training and testing:
```
python scripts/benchmarks.py

```


---
# Project Organization

    ├── data/              <- Data saved in this directory.
    │   ├── raw/           <- Contains raw data extracted from the MIMIC dataset.
    │   ├── processed/     <- Contains processed data organized as reusable modules for final dataset generation and other future tasks.
    │   └── final/         <- Contains the final dataset ready for model training.
    │
    ├── LICENSE            <- TODO
    │
    ├── models/            <- [TODO] Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks/         <- Jupyter notebooks.
    │
    ├── README.md  
    │
    ├── requirements.txt   <- [TODO] The requirements file for reproducing the analysis environment, e.g., 
    │                         generated with `pip freeze > requirements.txt` 
    │                      <- [We are using basic libraries that Colab already provides, but it is also important to save the current version for project reproducibility.]
    ├── scripts/           <- Contains specific task-oriented scripts by utilizing the functions and classes defined in the src directory.
    │   ├── cohort_extraction.py       <- Cohort extraction for critically ill trauma patients.
    │   ├── assign_sepsis_label.py     <- Assign Post-trauma Sepsis Label according to the definition.
    │   ├── generate_dataset.py        <- Generate dataset according to the Early Sepsis Onset Prediction Setup.
    │   └── benchmarks.py              <- Benchmarks: initialize model; training; evaluation.
    │           
    ├── setup.py           <- [TODO: file config?] Makes project pip installable (pip install -e .) so src can be imported.
    └──── src/               <- [TODO] Source code for use in this project. [Reusable modules, libraries, and functions that are essential for the project's operations.]
        ├── __init__.py    <- [TODO?] Makes src a Python module. [?? Not sure what should be included yet.]
        │
        ├── data           <- [TODO] Scripts to extract and preprocess raw data from MIMIC III.
        │
        ├── features       <- [TODO] Scripts to generate time-series data according to the Early Sepsis Onset Prediction Setup (including preprocessing, filling missing values, splitting, etc.) [this folder may merge with data folder].
        │
        └── models         <- [TODO] Scripts to train models and then use trained models to make predictions.
            │                 
            ├── predict_model.py    <- [Draft: will be improved later].
            └── train_model.py      <- [Draft: will be improved later].


    ## The following folders are in the template, but I am not sure what to include. I will add them as TODO for now.
    ├── docs               <- [TODO] Contains project documentation.
    ├── references         <- [TODO] Data dictionaries, manuals, and all other explanatory materials.  [maybe link to medical supporting documents?]
    │
    ├── reports            <- [TODO] Generated analysis as HTML, PDF, LaTeX, etc. [maybe supplementary and demographic reports?]
    │   └── figures        <- Generated graphics and figures to be used in reporting.
    ├── tests/             <- Test scripts.
    │   ├── test_data_loading.py <- [Test MIMIC III Data access].
    │   └── other_test.py

---

# Acknowledgements

[] TODO

LICENSE [TODO:add link]
