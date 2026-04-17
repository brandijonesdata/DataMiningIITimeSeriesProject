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
- `Final/timeimmtsfbjonesmortclassragradd.py`
- `Final/FINALDataMining2AnalyticsDay.pptx`
- `Final/TIME_IMM_TSF_Project_Methods.pdf`

## How to Run
Code for project and processed data files all from motivating paper git: `(https://github.com/blacksnail789521/Time-IMM)`
1. Install dependencies from `requirements.txt`
2. Run the main scripts from `Final/`

## Notes
This repository includes archived duplicate versions created during development. The recommended files to review are the ones in Final folder.
@inproceedings{
chang2025timeimm,
title={Time-{IMM}: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series},
author={Ching Chang and Jeehyun Hwang and Yidan Shi and Haixin Wang and Wei Wang and Wen-Chih Peng and Tien-Fu Chen},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2025},
url={https://openreview.net/forum?id=yeqrrn51TL}
}
