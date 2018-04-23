# train and evaluate multinomial naive bayes classifiers with a variety   
# of bag-of-words feature sets to benchmark each datasets using sklearn 

from __future__ import print_function

import csv
import sklearn.feature_extraction.text
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
import numpy as np
import time

print("loading data")

training_data = {}

# paths for each dataset
training_data['sent140_100k'] = ''
training_data['SemEval'] = ''
training_data['sent140Full'] = ''
training_data['amazon'] = ''
training_data['yelp'] = ''

out_file = open("output_datasets_u.csv", 'w')
print("dataset,ngrams,num_features,accuracy", file=out_file)
  
for training_set in training_data:
  print(training_set)
  data = []
  with open(training_data[training_set]) as f:
    csv_file = csv.reader(f)
    count = 0
    for line in csv_file:
        count +=1
        data.append((line[0].decode('latin-1').encode("utf-8"),line[1]))   

  text, labels = zip(*data)
  clf = MultinomialNB()

  train_split = int(len(text)*.9)
  test_split= train_split+int(len(text)*.1)

  print("making vectorizer")
  vectorizer = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(1,1))
  print("fitting vectorizer")
  vectorizer.fit(text)
  print("transforming text")
    
  text_ngram = vectorizer.transform(text) 
  labels = np.asarray(labels)

  for i in range(5):
    p = np.random.permutation(len(labels))
    text_ngram = text_ngram[p]
    labels = labels[p]

    clf.fit(text_ngram[0:train_split], labels[0:train_split])
    print("scoring model")
    run_score = clf.score(text_ngram[train_split:test_split], labels[train_split:test_split])
    print(training_set,",",text_ngram.shape[1],",",run_score)
    print(training_set,",",text_ngram.shape[1],",",run_score, file=out_file)
