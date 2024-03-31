# Max Todd
# Applied Machine Learning

import pandas as pd
import os.path
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re


def removeStopWords(text):
    '''
    Remove stop words from the text to reduce bias for them
    ex. "is", "the", "and", etc.

    Parameters
    ----------
    text: str
     Text to remove stop words from

    Returns
    -------
    str
     text parameter without stop words
    '''

    # error check - return empty string on non string
    if type(text) is not str:
        return ''

    # remove stop words
    noStopWords = ''
    words = str(text).split()
    for word in words:
        if word not in stopwords.words('english'):
            noStopWords += word + ' '

    return noStopWords.strip()


def normalizeText(text):
    '''
    Remove unecessary characters and normalize to reduce variety
    as much as possible across different messages

    Parameters
    ----------
    text: str
     Text to normalize

    Returns
    -------
    str
     normalized text parameter
    '''
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text))
    text = re.sub(r'\s+', ' ', str(text)).lower()
    text = re.sub(r'[^\w\s]', '', str(text))
    text = re.sub(r'\s+', ' ', str(text)).strip()
    return text


def tokenizeText(text):
    '''
    Tokenize the text

    Parameters
    ----------
    text: str
     Text to tokenize

    Returns
    -------
    str
     tokenized text parameter
    '''
    tokens = word_tokenize(str(text))
    return tokens


def loadDataset(path, textColumnName, saveName, removeCols=[]):
    '''
    Load the dataset and return a new dataframe containing the
    preprocessed text data. Saves into ./data/preprocessed/ex.csv

    Parameters
    ----------
    path: str
     path of the CSV file to load the dataset from
    textColumnName: str
     name of the column to perfrom the text preprocessing to
    saveName: str
     name of the csv file to save into ./data/preprocessed/saveName.csv
    removeCols: [str]
     List of column names of the df loaded from path to drop

    Returns
    -------
    pandas.Dataframe
     dataframe of the loaded data from csv file inputted and preprocessed text
    '''

    # Download nltk dependencies if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')

    # Make sure that the file is installed
    if not os.path.isfile(path):
        print(f'Error: {path} does not exist')
        return None

    # Load df
    df = pd.read_csv(path, encoding='unicode_escape')

    # Drop unwanted columns
    if len(removeCols) > 0:
        df.drop(columns=removeCols, inplace=True)

    # Drop nil / null values
    df.dropna(inplace=True)

    # Turn sentiment into integer
    df.loc[df['sentiment'] == 'negative', 'sentiment'] = int(0)
    df.loc[df['sentiment'] == 'neutral', 'sentiment'] = int(2)
    df.loc[df['sentiment'] == 'positive', 'sentiment'] = int(4)

    # Preprocess the data with NLP
    print('\tNormalizing...')
    df['normalizedText'] = df[textColumnName].apply(normalizeText)
    print('\tRemoving stop words...')
    df['normalizeNoStop'] = df['normalizedText'].apply(removeStopWords)
    print('\tTokenizing...')
    df['tokenized'] = df['normalizeNoStop'].apply(tokenizeText)

    # remove all last nan to be safe
    df.dropna(inplace=True)

    # Save to file
    Path('./data/preprocessed').mkdir(parents=True, exist_ok=True)
    df.to_csv('./data/preprocessed/' + saveName + '.csv', index=False)

    return df


# Modify and save datasets
if __name__ == '__main__':

    print('Working on training dataset...')
    loadDataset(path='./data/original/train.csv',
                textColumnName='text',
                saveName='train',
                removeCols=[
                    'textID',
                    'selected_text',
                    'Time of Tweet',
                    'Age of User',
                    'Country',
                    'Population -2020',
                    'Land Area (Km²)',
                    'Density (P/Km²)'
                ])

    print('\nWorking on testing dataset...')
    loadDataset(path='./data/original/test.csv',
                textColumnName='text',
                saveName='test',
                removeCols=[
                    'textID',
                    'Time of Tweet',
                    'Age of User',
                    'Country',
                    'Population -2020',
                    'Land Area (Km²)',
                    'Density (P/Km²)'
                ])
