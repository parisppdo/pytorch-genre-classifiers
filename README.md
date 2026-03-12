# 🎶 Music Genre Classification (Using Pytorch)

## Table of Contents
- [📖 Overview](#-overview)
- [🗂️️ Project Structure](#️-project-structure)
- [🛠️ Data Preparation](#️-data-preparation)
- [⚠️ Disclaimers](#️-disclaimers)

## 📖 Overview

This repository contains the experimental work and supporting material
for my **Master’s thesis**, focusing on **music genre classification
using deep learning techniques**.

The project includes a series of experiments conducted on the GTZAN dataset,
exploring different deep learning approaches implemented in PyTorch
to automatically recognize and categorize audio tracks into ten genres:
Blues, Classical, Country, Disco, HipHop, Jazz, Metal, Pop, Reggae, and Rock.

The goal of this work is to study and compare the performance of various
neural network architectures and audio feature representations,
providing insights into their effectiveness for music genre classification.
The experiments progressively refine the models and classifiers
developed as part of the thesis.

---

## 🗂️ Project Structure

The folder structure of the project is the following:  
```
MusicGenreClassification/
├──AST-Tuned Mel Spectrogram Classifiers/
│ ├── 3 segment division/
│ └── 10 segment division/
├── Mel Spectrogram Classifiers/
└── MFCC Classifiers/
```

Each folder contains the corresponding notebook files for each architecture 
(MLP, CNN, etc.). Each notebook documents the experimental procedure carried 
out for its respective architecture. Each notebook follows the same pipeline and
includes notes on the whole procedure.

In our experiments, we aim to compare the performance of each architecture under different conditions:
* Audio features represented as MFCCs or Mel Spectrograms
* Audio features with higher frequency resolution versus higher time resolution
* Ιnput data with longer duration but fewer samples, or shorter duration but more samples

The comparisons described above are carried out in experiments included in each folder.
* __AST-Tuned Mel Spectrogram Classifiers__: This folder includes Mel Spectrograms as
input data. The input data have higher time resolution.
    * __3 segment division__: This folder contains input data derived from 
    10-second audio samples. The whole dataset size is 3000 samples.
    * __10 segment division__: This folder contains input data derived from 
    3-second audio samples. The whole dataset size is 10000 samples.
* __Mel Spectrogram Classifiers__: This folder includes Mel Spectrograms as
input data. The input data have higher frequency resolution. 
The whole dataset is 9996 samples.
* __MFCC Classifiers__: This folder includes MFCCs as
input data. The input data have higher frequency resolution. 
The whole dataset is 9996 samples.

---

## 🛠️ Data Preparation

The dataset has 100 music samples of 30 sec for each genre. 
An initial processing of the data has been implemented in the
`preprocess.py` file of each classifier group.

* In the **Mel Spectrogram Classifiers** folder, `preprocess.py` extracts **Mel Spectrogram features**.
* In the **MFCC Classifiers** folder, `preprocess.py` extracts **MFCC features** from each audio segment.
* In the **AST-Tuned Mel Spectrogram Classifiers** folder, `preprocess_with_AST_options.py`, `preprocess_AST_extractor_to_pt.py`
and `preprocess_original.py` extract **Mel Spectrogram features**.

Each script converts the audio files into tensors and organizes them 
into a `.json` dictionary that contains:
- the names of the 10 genre classes,
- the feature tensors (MFCCs or Mel Spectrograms),
- the corresponding labels. 

In the case of ASTClassifier models, tensors are organized into `.pt` files due to file 
size limitations.

Each sound file is divided into 10 segments to increase the number of 
available data. In the case of **AST-Tuned Mel Spectrogram Classifiers** we include
a 3 segmentation division.
For the Mel Spectogram, we use 64 Mel Banks.    
For MFCCs, we use 13 coefficients, which is a typical number used in 
music genre classification.

A small `.json` file named `mfcc_data_small.json` is included in the main directory 
as an example of the generated data structure.  This is an MFCC example and in 
the case of Mel Spectrograms, the file is similar.
(The full `.json` files are not included due to GitHub’s file size limitations.)

The extracted `.json` file is loaded in each *classifier* 
file for further processing (e.g. `CNNClassifier.ipynb`). 
The pipeline is described within
the notebook file.

---

## ⚠️ Disclaimers

> The GTZAN dataset is **not** included in the repository for royalty reasons. You can 
> download it [here](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification),
> unzip it and include it in the main directory in a folder named *"genres_full"*.

> The `.json` files with the full feature data (MFCC or Mel Spectrogram)
> are not included due to their size. Instead, a small sample file 
> named `mfcc_data_small.json` is included in the main directory, to 
> demonstrate the data format and structure.  
> The `.ipynb` files have been run on the full dataset.
