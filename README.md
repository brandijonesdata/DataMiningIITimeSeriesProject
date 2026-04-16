# DataMiningIITimeSeriesProject

## Overview
This repository contains our Data Mining II time series project focused on irregular clinical time series modeling using MIMIC-based and FNSPID data and reproduction/extension of Time-IMM style methods.

## Objectives
- preprocess MIMIC-style admissions, outputs, and prescription data
- build mortality/classification pipelines for irregular time series
- compare baseline and modified gating/classification approaches
- document results in reports and presentation slides

## Repository Structure
- `Brandi/`: Brandi Python scripts
- `chad/`: Chad python scripts
- `nnenna/`: Nnenna python scripts
- `Notes/`: written reports and PDFs
- `Presentations/`: presentation decks
- `IntermediateFiles/`: csvs and other docs needed for project pipeline

## Main Files
- `Final/run_mimic_mortality_irregular_oldgate.py`
- `Final/run_mimic_classifier_oldgate.py`
- `Final/03_group_project.ipynb`
- `Final/TIME_IMM_Reproduction3.pdf`

## How to Run
1. Install dependencies from `requirements.txt`
2. Place required processed data files in `[data/processed/](https://github.com/blacksnail789521/Time-IMM)`
3. Run the main scripts from `Final/`

## Notes
This repository includes archived duplicate versions created during development. The recommended files to review are the ones in Final folder.
