# Max Todd
# Applied Machine Learning

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from time import time

# |%%--%%| <90Vp4XA67T|8wYOMPCEGf>

# Load the processed dataset (get from running preprocessing.py)
traindf = pd.read_csv('./data/preprocessed/train.csv',
                      encoding='unicode_escape')
testDF = pd.read_csv('./data/preprocessed/test.csv',
                     encoding='unicode_escape')

# |%%--%%| <8wYOMPCEGf|ZmqSIREtN8>

# remove nan
traindf.dropna(inplace=True)
testDF.dropna(inplace=True)

# |%%--%%| <ZmqSIREtN8|BlKHXm7fUo>

# Create storage for results
models = [
            'NB with NTD',
            'NB with TTD',
            'SVM with NTD',
            'SVM with TTD'
         ]

accuracies = [0 for i in range(len(models))]
trainTime = [0 for i in range(len(models))]

# |%%--%%| <BlKHXm7fUo|iSEF0piTip>
r"""°°°
Naive Bayes with normalized text data
Accuracy = 0.6293984108967083
°°°"""
# |%%--%%| <iSEF0piTip|J2LVgTkcTk>

# Start timing
train = time()

# Make and train Naive Bayes model using the normalized text without stop words
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(traindf['normalizeNoStop'], traindf['sentiment'])

# Stop timing
train = time() - train

# Add time to results
trainTime[0] = train

# |%%--%%| <J2LVgTkcTk|8Fjyc2TC6S>

# Predict
predicted = model.predict(testDF['normalizeNoStop'])

# Get the accuracy using the normalized text without stop words
accuray = accuracy_score(predicted, testDF['sentiment'])
print(f'NB with normalized accuracy={accuray}')
print(f'time: {train * 1000} ms')

# Add accuracy to results
accuracies[0] = accuray

# |%%--%%| <8Fjyc2TC6S|r0I3f1kaN1>
r"""°°°
Naive Bayes with tokenized text data
Accuracy = 0.6285471055618616
°°°"""
# |%%--%%| <r0I3f1kaN1|a5OQ7lKu9p>

# Start timing
train = time()

# Make and train Naive Bayes model using the tokenized text
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(traindf['tokenized'], traindf['sentiment'])

# Stop timing
train = time() - train

# Add time to results
trainTime[1] = train

# |%%--%%| <a5OQ7lKu9p|DXq8qypW7x>

# Predict
predicted = model.predict(testDF['tokenized'])

# Get the accuracy using the tokenized text
accuray = accuracy_score(predicted, testDF['sentiment'])
print(f'NB with tokenized accuracy={accuray}')
print(f'time: {train * 1000} ms')

# Add accuracy to results
accuracies[1] = accuray

# |%%--%%| <DXq8qypW7x|fT2ShfNN6w>
r"""°°°
SVM with normalized text data
Accuracy = 0.7017593643586834
°°°"""
# |%%--%%| <fT2ShfNN6w|g0WkFzfmMp>

# Start timing
train = time()

# Make and train Naive Bayes model using the normalized text without stop words
model = make_pipeline(TfidfVectorizer(), SVC())
model.fit(traindf['normalizeNoStop'], traindf['sentiment'])

# Stop timing
train = time() - train

# Add time to results
trainTime[2] = train

# |%%--%%| <g0WkFzfmMp|vnI5Nygdwv>

# Predict
predicted = model.predict(testDF['normalizeNoStop'])

# Get the accuracy using the tokenized text
accuray = accuracy_score(predicted, testDF['sentiment'])
print(f'SVM with normalized text data accuracy = {accuray}')
print(f'time: {train * 1000} ms')

# Add accuracy to results
accuracies[2] = accuray

# |%%--%%| <vnI5Nygdwv|j31xd1bEUi>
r"""°°°
SVM with tokenized text data
Accuracy = 0.7023269012485811
°°°"""
# |%%--%%| <j31xd1bEUi|2Zd1Pegj2t>

# Start timing
train = time()

# Make and train Naive Bayes model using the normalized text without stop words
model = make_pipeline(TfidfVectorizer(), SVC())
model.fit(traindf['tokenized'], traindf['sentiment'])

# Stop timing
train = time() - train

# Add time to results
trainTime[3] = train

# |%%--%%| <2Zd1Pegj2t|RgursE1VMN>

# Predict
predicted = model.predict(testDF['tokenized'])

# Get the accuracy using the tokenized text
accuray = accuracy_score(predicted, testDF['sentiment'])
print(f'SVM with tokenized text data = {accuray}')
print(f'time: {train * 1000} ms')

# Add accuracy to results
accuracies[3] = accuray
