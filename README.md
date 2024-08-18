# Rare Event Early Detection: Sepsis Onset for Critically Ill Trauma Patients

### Overview
This repository contains the code for generating a dataset aimed at detection the early onset of hospital-acquired sepsis among critically ill patients within a trauma cohort from the MIMIC-III dataset.

The implementation is fully in Python, following the methodology described in the paper "Rare Event Early Detection: Sepsis Onset for Critically Ill Trauma Patients." The project leverages the Google BigQuery server to access and manage the MIMIC-III database.

### Dataset
We introduce a publicly available, reliable dataset derived from the MIMIC-III database. This dataset encompasses a well-defined trauma cohort with corresponding sepsis labels and onset timestamps, specifically targeting the task of early hospital-acquired sepsis onset detection among critically ill trauma patients. The dataset includes comprehensive data analysis and medical context for both the cohort and the sepsis labels, along with well-structured time-series vital sign features suited for ICU deployment.

---

# Usage

The project includes several notebooks and scripts to perform different tasks:

### Get Started 
A detailed version of this README is available in notebook form at `notebooks/ReadMe.ipynb`. The project is expected to run on Google Colab, as it uses the Google BigQuery server to access and manage the MIMIC-III database.

### Check MIMIC Dataset Access
To get or verify access to the MIMIC-III dataset through Google BigQuery, please refer to `notebooks/MIMIC-III_Data_Access_Instructions.ipynb`.

### Cohort Extraction: Critically Ill Trauma Patients
This section extracts a cohort of critically ill trauma patients from the MIMIC-III v1.4 dataset, applying stringent criteria such as age range, admission duration, and mechanical ventilation days. The refined cohort, tailored for early sepsis onset detection, consists of 1,570 admissions. For more details, see `notebooks/Cohort_Extraction.ipynb`.

```python
from scripts.cohort_extraction import extract_trauma_cohort_ids

# Extract cohort IDs and generate a statistics report
trum_ids = extract_trauma_cohort_ids(project_path_obj,    # Saved File Paths
                                     PROJECT_ID,          # To query raw data
                                     is_report=True,      # Print statistics report
                                     is_saved=True        # Save the cohort IDs
                                    )
```

### Assign Post-Trauma Sepsis Labels
This section provides an overview of the process used to assign sepsis labels and determine onset times for patients within the critically ill trauma cohort. The methodology adheres to Sepsis-3 guidelines, with a focus on identifying suspected infections and associated organ dysfunction. This process is essential for accurately labeling Post-Trauma sepsis onset, which is crucial for subsequent analysis and model development. For detailed implementation, refer to `notebooks/Sepsis_Onset_Label_Assignment.ipynb`.

Out of the 1,570 trauma admissions analyzed, 729 had suspected infections, and 535 were confirmed with sepsis. The peak of sepsis onset typically occurs on the 5th day after hospital admission, as illustrated in the graph in the output of the following code.

```python
from scripts.sepsis_onset_label_assignment import assign_sepsis_labels

# Assign sepsis labels and onset times for each patient in the cohort
sepsis_label_df = assign_sepsis_labels(project_path_obj,  # Saved File Paths
                                       PROJECT_ID         # To query raw data
                                      )
```

### Generate Dataset
In this section, you will generate the Post-Traumatic Sepsis dataset, available in both versions with and without missing values, derived from the MIMIC-III v1.4 dataset. For a detailed explanation of the dataset construction process, refer to `notebooks/Early_Sepsis_Onset_Detection_Setup.ipynb`.

The dataset includes the following columns:

- **Temporal Features**: Multivariate time-series input data, with dimensions (# of timestamps, # of features).
- **Label**: A binary value (0 or 1) representing the output label.
- **Dataset**: Indicates whether the instance belongs to the training or test set.

Each row in the dataset corresponds to one nighttime instance and includes patient identifiers (subject_id, hadm_id) and a timestamp ID (Date, Night).

```python
from scripts.early_sepsis_onset_detection_setup import dataset_construction

data_with_nan_df, data_wo_nan_df = dataset_construction(project_path_obj, PROJECT_ID, is_report=True)
```

### Benchmarks

#### 1. **Simple Solutions for Datasets Without Missing Values**

This pipeline focuses on the Post-Traumatic Sepsis dataset **without missing values**, targeting rare event early detection, specifically sepsis onset.

**Key Techniques:**
- **Reweighting**: Adjusting class weights to handle class imbalance.
- **Resampling**: Balancing the dataset through over-sampling or under-sampling techniques.
- **Augmentation**: Generating synthetic samples to enrich the dataset.

The pipeline evaluates the effectiveness and limitations of these methods in managing class imbalance, where positive instances constitute about 4% of the dataset. For more detailed explanations, refer to `notebooks/ML_Pipeline_No_Missing.ipynb`.

#### Performance Comparison of Different Methods (Average of 5-Fold Cross-Validation)
| Method                    | Precision | Recall   | F1_score | AUC (PR) | Specificity | TN    | FP    | FN    | TP    |
|---------------------------|-----------|----------|----------|----------|-------------|-------|-------|-------|-------|
| Simple Classification      | 0.0       | 0.0      | 0.0      | 0.569237 | 1.0         | 1660.6| 0.0   | 73.4  | 0.0   |
| Reweighting                | 0.048253  | 0.458521 | 0.086815 | 0.556912 | 0.596183    | 986.2 | 674.4 | 39.8  | 33.6  |
| Augmentation: TimeWarp     | 0.0       | 0.0      | 0.0      | 0.555446 | 1.0         | 1660.6| 0.0   | 73.4  | 0.0   |
| Undersampling + Oversampling | 0.050142 | 0.441644 | 0.089228 | 0.541260 | 0.618681    | 1020.8| 639.8 | 41.0  | 32.4  |
| SMOTE                      | 0.041939  | 0.346630 | 0.074615 | 0.516961 | 0.649277    | 1076.0| 584.6 | 48.0  | 25.4  |
| SMOTE + TimeWarp           | 0.044556  | 0.444968 | 0.080578 | 0.522805 | 0.577693    | 953.2 | 707.4 | 40.8  | 32.6  |



#### 2. **Masked Autoencoder for Datasets with Missing Values**

This pipeline addresses the challenges posed by **missing data** and class imbalance in the Post-Traumatic Sepsis dataset.

**Approach:**
- **Masked Autoencoder (MAE)**: Pre-trains on data with missing values, learning robust feature representations and serving as data augmentation.
- **Classifier**: Applied after MAE pre-training for sepsis onset detection.

This pipeline effectively handles missing values and the significant class imbalance, where positive samples represent approximately 4% of the dataset. Detailed explanations can be found in `notebooks/ML_Pipeline_with_Missing.ipynb`.

#### Cross-Validation Metrics Summary

| Fold | Num Train Samples | Num Train Patients | Num Val Samples | Num Val Patients | Precision | Recall  | F1 Score | AUC (PR) | Specificity | TN  | FP  | FN  | TP  | Reconstruct Error |
|------|-------------------|--------------------|-----------------|------------------|-----------|---------|----------|-----------|-------------|------|------|------|------|------------------|
| 1    | 14,158             | 1,085              | 1,923           | 272              | 0.047252  | 0.620253 | 0.087814 | 0.561499  | 0.464208    | 856  | 988  | 30   | 49   | 6.369861          |
| 2    | 14,178             | 1,085              | 1,913           | 272              | 0.043912  | 0.556962 | 0.081406 | 0.516813  | 0.477644    | 876  | 958  | 35   | 44   | 6.655287          |
| 3    | 14,206             | 1,086              | 1,898           | 271              | 0.043197  | 0.512821 | 0.079681 | 0.538955  | 0.513187    | 934  | 886  | 38   | 40   | 6.600450          |
| 4    | 14,446             | 1,086              | 1,779           | 271              | 0.047091  | 0.645570 | 0.087780 | 0.556750  | 0.392941    | 668  | 1,032 | 28   | 51   | 6.436116          |
| 5    | 14,396             | 1,086              | 1,803           | 271              | 0.047410  | 0.692308 | 0.088743 | 0.580654  | 0.371014    | 640  | 1,085 | 24   | 54   | 6.519011          |
| **Average** | **14,276.8**     | **1,085.6**          | **1,863.2**         | **271.4**        | **0.045772** | **0.605583** | **0.085085** | **0.550934** | **0.443799**  | **794.8** | **989.8** | **31.0** | **47.6** | **6.516145**        |

---

# Project Organization

    ├── data/              <- Data saved in this directory.
    │   ├── raw/           <- Contains raw data extracted from the MIMIC dataset.
    │   ├── processed/     <- Contains processed data organized as reusable modules for final dataset generation and other future tasks.
    │
    ├── dataset/           <- Contains the final dataset ready for model training.
    │
    ├── LICENSE            <- TODO
    │
    ├── notebooks/         <- Jupyter notebooks matching with scripts
    │   ├── MIMIC-III Data Access Instructions.ipynb      <- Instructions on how to access the MIMIC-III dataset.
    │   ├── cohort_extraction.ipynb                       <- Cohort extraction for critically ill trauma patients.
    │   ├── Sepsis_Onset_Label_Assignment.ipynb           <- Assign Post-trauma Sepsis Onset Label according to the definition.
    │   ├── Early_Sepsis_Onset_Detection_Setup.ipynb      <- Generate dataset according to the Early Sepsis Onset Prediction Setup.
    │
    ├── README.md  
    │
    ├── requirements.txt   <- Basic libraries used in the project (most available on Colab); saved for project reproducibility.
    ├── scripts/           <- Task-oriented scripts utilizing functions and classes defined in the src directory.
    │   ├── cohort_extraction.py                       <- Cohort extraction for critically ill trauma patients.
    │   ├── Sepsis_Onset_Label_Assignment.py           <- Assign Post-trauma Sepsis Onset Label according to the definition.
    │   ├── Early_Sepsis_Onset_Detection_Setup.py      <- Generate dataset according to the Early Sepsis Onset Prediction Setup.
    │
    ├── src/               <- Source code for use in this project; reusable modules, libraries, and functions essential for the project's operations.
        ├── __init__.py    
        │
        ├── data          
    └── supplementary/
        ├── qualified_traumatic_ICD9_Ecodes.xlsx <- Qualifying ICD-9 E codes for the trauma cohort.

