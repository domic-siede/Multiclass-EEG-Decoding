# ER_2025 - Multiclass EEG Decoding of Emotional Reactivity and Cognitive Reappraisal

MATLAB pipeline for multiclass EEG decoding of three emotion-regulation conditions using ROI-based oscillatory features and linear SVM classification within an ECOC framework.

## Overview

This repository contains the analysis code used to classify three affective/regulatory states from single-trial EEG data:

- **0 = Neutral viewing**
- **1 = Natural negative viewing**
- **2 = Reappraisal of negative stimuli**

The pipeline extracts **log-transformed bandpower** features from predefined scalp **regions of interest (ROIs)** and evaluates decoding performance using **repeated within-subject stratified K-fold cross-validation**, **majority voting**, and a **within-subject permutation test**.

In addition to the main decoding analysis, the code includes:

- ROI × band subset scouting
- within-subject effect size estimation for top-performing subsets
- descriptive ROI-level power spectral density (PSD) plots
- descriptive ROI-level spectrograms

## Feature Extraction

Features are computed from the stimulus window:

- **0.30-3.80 s**

Using Welch’s method, the script extracts log10 bandpower for the following frequency bands:

- **Theta:** 4-8 Hz
- **Alpha:** 9-12 Hz
- **Beta:** 15-30 Hz

### Regions of Interest (ROIs)

- **Frontal (F):** Fz, FCz, FC1, FC2, F1, F2  
- **Central (C):** C3, C1, Cz, C2, C4  
- **Parietal (P):** P3, Pz, P4, POz  
- **Occipital (O):** O1, Oz, O2  
- **Temporal (T):** T7, T8  

## Classification Pipeline

The main model is a **multiclass linear Support Vector Machine (SVM)** implemented through an **Error-Correcting Output Codes (ECOC)** framework.

### Validation strategy

- repeated **within-subject stratified K-fold cross-validation**
- **20 repetitions**
- **majority voting** across repetitions
- class weighting to reduce imbalance effects

### Main performance metrics

- **Balanced Accuracy**
- **Macro-F1**
- **Cohen’s Kappa**

## Permutation Testing

To assess statistical significance, the script performs a **within-subject label-shuffled permutation test**:

- **1000 permutations**
- labels shuffled **within participant**
- preserves subject-level trial structure
- compares observed balanced accuracy against the empirical null distribution

## Additional Analyses

### 1. ROI × Band Subset Scouting
The pipeline evaluates predefined feature subsets, including:

- all features
- theta across ROIs
- alpha across ROIs
- beta across ROIs
- all bands within each ROI
- selected single ROI-band combinations

### 2. Effect Size Estimation
For top-performing subsets, the script computes:

- **within-subject Hedges g**
- **bootstrap confidence intervals**
- **FDR-corrected p-values**

### 3. Descriptive Spectral Visualizations
The script also generates:

- ROI-averaged **PSD plots** (1-40 Hz)
- ROI-level **spectrograms** (1-40 Hz)
- PSD by condition for each ROI

## Requirements

- **MATLAB 2018b** or compatible version
- **EEGLAB**
- **FieldTrip**

## Input Data

The script expects preprocessed EEG datasets in `.set` format.

You must update the paths in the script before running it:

- `eeglabPath`
- `ftPath`
- `dataPath`

## Running the Analysis

Run the main MATLAB script after setting the correct toolbox and data paths.

The script will:

1. load all `.set` files
2. apply baseline correction
3. extract ROI × band features trial-by-trial
4. run multiclass decoding
5. compute subject-level and global metrics
6. perform permutation testing
7. run subset scouting and effect size analyses
8. generate PSD and spectrogram visualizations

## Output

The script reports results in the MATLAB console, including:

- subject-level decoding performance
- global decoding metrics
- confusion matrix
- permutation-test results
- sorted subset-scouting table
- effect sizes for top subsets

It also generates figures for:

- confusion matrix
- permutation histogram
- ROI PSD
- ROI spectrograms
- PSD by condition

## Citation

If you use this code, please cite the associated manuscript:

**Domic-Siede, M., & Calderón, C.**  
*Distinct Oscillatory Signatures of Emotional Reactivity and Cognitive Reappraisal Revealed Through Multiclass EEG Decoding.*

## Notes

This repository focuses on the decoding pipeline described in the manuscript and does **not** include unrelated exploratory feature modes. The current implementation is specifically aligned with the paper’s final ROI-based oscillatory feature space.

## Contact

**Marcos Domic-Siede**  
Escuela de Psicología, Universidad Católica del Norte  
Antofagasta, Chile
