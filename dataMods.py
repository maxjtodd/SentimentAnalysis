# Max Todd
# Data preprocessing

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

nltk.download('stopwords')
nltk.download('punkt')

# |%%--%%| <NNWb8Bmwuh|QLsNNn3CEy>
r"""°°°
Explore the Dataset
°°°"""
# |%%--%%| <QLsNNn3CEy|A7C1Hp4RzW>

# Load the dataset
trainDF = pd.read_csv('./data/original/train.csv', encoding='unicode_escape')
testDF = pd.read_csv('./data/original/train.csv', encoding='unicode_escape')

# Preview the dataset
trainDF.head()

# |%%--%%| <A7C1Hp4RzW|yR5WGXhvIZ>

# Print dataset information
print('Training:')
trainDF.info()
print('\n')
print(f'Test same schema, length = {len(testDF[list(testDF.columns)[0]])}')

# |%%--%%| <yR5WGXhvIZ|yaiVTg5eGw>
r"""°°°
Modify the datset
°°°"""
# |%%--%%| <yaiVTg5eGw|kPy8CgUia2>


def removeStopWords(text):
    '''
    Remove stop words from the text to reduce bias for them
    ex. "is", "the", "and", etc.
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
    '''
    tokens = word_tokenize(str(text))
    return tokens

# |%%--%%| <kPy8CgUia2|Vheb64xcfk>


def loadDataset(path, textColumnName):
    '''
    Load the dataset and return a new dataframe containing the
    preprocessed text data

    Parameters
    ----------
    path: str
     path of the CSV file to load the dataset from
    textColumnName: str
     name of the column to perfrom the text preprocessing to

    Returns
    -------
    pandas.Dataframe
     dataframe of the loaded data from csv file inputted and preprocessed text
    '''

    # Load df
    df = pd.read_csv(path, encoding='unicode_escape')

    # Drop nil / null values
    df.dropna(inplace=True)

    # Preprocess the data with NLP
    print('Normalizing...')
    df['normalizedText'] = df[textColumnName].apply(normalizeText)
    print('Removing stop words...')
    df['normalizeNoStop'] = df['normalizedText'].apply(removeStopWords)
    print('Tokenizing...')
    df['tokenized'] = df['normalizeNoStop'].apply(tokenizeText)

    return df


# |%%--%%| <Vheb64xcfk|k2syakQBc1>
r"""°°°
You can save the dataset obtained from the above function to use for models,
or continue from here.
°°°"""
# |%%--%%| <k2syakQBc1|G0VOa28MzF>
