# DataMiningIITimeSeriesProject

## Overview
This repository contains our Data Mining II time series project focused on irregular clinical time series modeling using MIMIC-based and FNSPID data and reproduction/extension of Time-IMM style methods.

## Objectives
- preprocess MIMIC-style admissions, outputs, and prescription data
- build mortality/classification pipelines for irregular time series
- compare baseline and modified gating/classification approaches
- document results in reports and presentation slides

## Repository Structure
- `src/`: main Python scripts
- `notebooks/`: exploratory and final analysis notebooks
- `data/processed/`: processed input files used by notebooks/scripts
- `reports/`: written reports and PDFs
- `slides/`: presentation decks
- `archive/`: older or duplicate versions retained for record-keeping

## Main Files
- `src/run_mimic_mortality_irregular_oldgate.py`
- `src/run_mimic_classifier_oldgate.py`
- `notebooks/03_group_project.ipynb`
- `reports/TIME_IMM_Reproduction3.pdf`

## How to Run
1. Install dependencies from `requirements.txt`
2. Place required processed data files in `data/processed/`
3. Run the main scripts from `src/`

## Notes
This repository includes archived duplicate versions created during development. The recommended files to review are the ones listed above.
