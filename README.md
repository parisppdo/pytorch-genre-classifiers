# 🎶 Music Genre Classification (Using Pytorch)

## Table of Contents
- [📖 Overview](#-overview)
- [🛠️ Data Preparation](#️-data-preparation)
- [⚠️ Disclaimers](#️-disclaimers)

## 📖 Overview

This project contains experiments with music genre classification
using the GTZAN dataset. The project explores different deep 
learning approaches in PyTorch to automatically recognize and 
categorize audio tracks into ten genres: Blues, Classical, 
Country, Disco, HipHop, Jazz, Metal, Pop, Reggae, and Rock. 
This experimentation, provides insights on how various 
models perform on this task and progressively refine 
the classifiers.

---

## 🛠️ Data Preparation

The dataset has 100 music samples of 30 sec for each genre. 
An initial processing of the data has been implemented in the
`preprocess.py` file: Each sound file is divided into 10 
segments in order to increase the number of available data. 
An MFCC is extracted from each segment. The script saves
the data in a `.json` file that contains the name of the 
classes, the mfccs and the labels. Each mfcc is represented
by a 2D array of numerical values. The shape of this array 
is `(time_frame, mfcc_coefficient)`. We use 13 coefficients, 
which is a typical number used in music genre classification.

The extracted `.json` file is loaded in each *classifier* 
file for further processing (eg `CNNClassifier). 
The pipeline is described within
the notebook file.

---

## ⚠️ Disclaimers

> The GTZAN dataset is **not** included in the repository for royalty reasons. You can 
> download it [here](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification),
> unzip it and include it in the main directory in a folder named *"genres_full"*.

> The `.json` file with the mfcc data of the whole dataset is not
> included due to its size. Instead, a sample file named `data_med.json`
> is included. The `.ipynb` file has been run on the full dataset.