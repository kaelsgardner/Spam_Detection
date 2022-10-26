import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB


## 1 | Data Preprocessing ##
"""Prepare data before training"""

#Load dataset
dataset = pd.read_csv('Dataset/emails.csv')
print(f"Dataset head : \n{dataset.head()}\n")

#Check for and remove duplicates
dataset.drop_duplicates(inplace=True)

#Clean data, tokenizing it into words/tokens
def process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    cleaned = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return cleaned


#Convert the text into matrix of token counts
message = CountVectorizer(analyzer=process).fit_transform(dataset['text'])

#Split the data into training/testing 
X_train, X_test, y_train, y_test = train_test_split(message, dataset['spam'], test_size=0.20, random_state=0)

# Create and train the Naive Bayes Classifier
classifier = MultinomialNB().fit(X_train, y_train)

# Classifiers prediction
y_pred = classifier.predict(X_test)
print(f"Prediction results (y_pred): \n{y_pred}\n")

## Model Evaluation

print(f"Classification report :\n{classification_report(y_train, y_pred)}\n")
print(f"Confusion Matrix :\n{confusion_matrix(y_train, y_pred)}\n")
print(f"Model accuracy : {round(accuracy_score(y_train, y_pred), 2)*100} %")
