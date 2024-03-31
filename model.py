# Max Todd
# Applied Machine Learning

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# |%%--%%| <tECa0hJTnV|8wYOMPCEGf>

# Load the processed dataset (get from running preprocessing.py)
traindf = pd.read_csv('./data/preprocessed/train.csv', encoding='unicode_escape')
testDF = pd.read_csv('./data/preprocessed/test.csv', encoding='unicode_escape')

# |%%--%%| <8wYOMPCEGf|ZmqSIREtN8>

# remove nan
traindf.dropna(inplace=True)
testDF.dropna(inplace=True)

# |%%--%%| <ZmqSIREtN8|J2LVgTkcTk>

# Make and train model using the normalized text without stop words
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(traindf['normalizeNoStop'], traindf['sentiment'])

# Predict
predicted = model.predict(testDF['normalizeNoStop'])
# |%%--%%| <J2LVgTkcTk|8Fjyc2TC6S>

# Get the accuracy using the normalized text without stop words
accuray = accuracy_score(predicted, testDF['sentiment'])
print(f'{accuray}')

# |%%--%%| <8Fjyc2TC6S|a5OQ7lKu9p>

# Make and train model using the tokenized text
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(traindf['tokenized'], traindf['sentiment'])

# Predict
predicted = model.predict(testDF['tokenized'])

# |%%--%%| <a5OQ7lKu9p|DXq8qypW7x>

# Get the accuracy using the tokenized text
accuray = accuracy_score(predicted, testDF['sentiment'])
print(f'{accuray}')
