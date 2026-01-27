# 🎶 Music Genre Classification (Using Pytorch)

## Table of Contents
- [📖 Overview](#-overview)
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

The project includes two main classifier groups:
- **Mel Spectrogram Classifiers**
- **MFCC Classifiers**

Each group contains separate notebook files implementing different 
deep learning architectures (MLP, CNN, RNN-LSTM-GRU, etc.) for genre 
classification using the respective feature representation.

---

## 🛠️ Data Preparation

The dataset has 100 music samples of 30 sec for each genre. 
An initial processing of the data has been implemented in the
`preprocess.py` file of each classifier group:

- In the **MFCC Classifiers** folder, `preprocess.py` extracts **MFCC features** from each audio segment.
- In the **Mel Spectrogram Classifiers** folder, `preprocess.py` extracts **Mel Spectrogram features**.

Each script converts the audio files into tensors and organizes them 
into a `.json` dictionary that contains:
- the names of the 10 genre classes,
- the feature tensors (MFCCs or Mel Spectrograms),
- and the corresponding labels.

Each sound file is divided into 10 segments to increase the number of 
available data.  
For the Mel Spectogram, we use 64 Mel Banks.    
For MFCCs, we use 13 coefficients, which is a typical number used in 
music genre classification.     



A small `.json` file named `data_small.json` is included in each folder 
as an example of the generated data structure.  
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
> are not included due to their size. Instead, small sample files 
> named `data_small.json` are included in each classifier folder to 
> demonstrate the data format and structure.  
> The `.ipynb` files have been run on the full dataset.
