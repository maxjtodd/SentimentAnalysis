# Overview

The following is a sentiment analysis project. The dataset is a dataset full of tweets that are categorized as
either positive, neutral, or negative.

As of now, the model has a 0.629398 accuracy with the Naive Bayes model.

# Installing Dataset
## Via kaggle API
If you have the kaggle API downloaded, or if you activate the conda environment included in env.txt (instructions below), 
you can download the dataset by performing the following:
```sh
$ kaggle datasets download -d abhi8923shriv/sentiment-analysis-dataset
```
[how to set up API key](https://fcpython.com/extra-time/searching-downloading-kaggle-datasets-command-line)

## Via the Web: 
[Dataset page](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset/data)

# Set Up (reccomended)
1. Download the conda environment using: 
```sh
$ conda create --name <env name you want> --file <env.txt>
```
2. Activate the environment
```sh
$ conda activate <env name you want>
```
3. Move the data into the ./data/original directory
4. Run preprocessing.py to perform necessary data preprocessing steps
```sh
$ python preprocessing.py
```
5. Free to use model.ipynb at this point

## Manual
- Download the dataset to desired location
- dataMods.ipynb has the preprocessing code needed, can save wherever you want from that point
