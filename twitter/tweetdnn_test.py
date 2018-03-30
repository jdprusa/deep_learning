from __future__ import print_function

import keras
from keras.models import load_model
from keras import backend as K

import load_text
import numpy as np
import os

#parameters
batch_size = 100
num_classes = 2
model_dir = ""


def generator(data, max_len, batch_size):
  while True:
    text, labels_, labels, chunks = data.next_batch(batch_size, max_len, max_len, 'chop', shuffle=False)
    yield text, keras.utils.to_categorical(labels, num_classes)
    
def conv_accuracy(test, model, max_len, stride, steps_per_epoch):
  total = 0
  correct = 0
  total_2 = 0
  correct_2 = 0
  for step in range(steps_per_epoch+1):
    if step < steps_per_epoch:
      batch, labels_, labels, chunks = test.next_batch(batch_size, max_len, stride, 'conv', shuffle=False)
    else:
      if test._num_examples-steps_per_epoch*batch_size == 0:
        return correct/total
      batch, labels_, labels, chunks = test.next_batch(test._num_examples-steps_per_epoch*batch_size, max_len, stride, 'conv', shuffle=False)
    results = model.predict_on_batch(batch)
    results = np.split(results,chunks[:len(chunks)-1],axis=0)

    # average
    for i in range(len(results)):
      if type(results[i]) != type(np.array([1])):
        print('something is wrong')
      results[i] = np.sum(results[i], axis=0)/len(results[i])
      total += 1
      if results[i][labels[i]] > 0.5:
        correct +=1
            
  return correct/total
  
def conv_output(test, model, max_len, stride, steps_per_epoch, scenario):
  total = 0
  correct = 0
  total_2 = 0
  correct_2 = 0

  for step in range(steps_per_epoch+1):
    if step < steps_per_epoch:
      batch, labels_, labels, chunks = test.next_batch(batch_size, max_len, stride, 'conv', shuffle=False)
    else:
      if test._num_examples-steps_per_epoch*batch_size == 0:
        return 
      batch, labels_, labels, chunks = test.next_batch(test._num_examples-steps_per_epoch*batch_size, max_len, stride, 'conv', shuffle=False)
    results = model.predict_on_batch(batch)
    results = np.split(results,chunks[:len(chunks)-1],axis=0)

    # average
    with open(scenario+"_conv.csv", 'a') as out:
      for i in range(len(results)):
        if type(results[i]) != type(np.array([1])):
          print('something is wrong')
        results[i] = np.sum(results[i], axis=0)/len(results[i])
        print(str(results[i][0])+","+str(results[i][1])+","+str(labels[i]), file=out)

def main():
  print("train_set, max_len, test_set, accuracy, conv_accuracy")
    #model_name = "amazon0150.h5"
  for model_name in os.listdir(model_dir):
    model_len = len(model_name)
    max_len = int(model_name[model_len-7:model_len-3])
    
    #load model 
    model = load_model(model_dir+model_name)
    #evaluate on test data
  
    # test sets
    test_sets = {}
    test_sets['SemEval'] = ""
    test_sets['amazon'] = ""
    test_sets['yelp'] = ""
    test_sets['tweet'] = ""
  
    # load data
    for test_set in test_sets:
      model_name_short = model_name[:model_len-7]
      #print(model_name_short+",",str(max_len)+",", test_set, end=",")  #comment out if getting probabilities
      print(model_name_short+",",str(max_len)+",", test_set)            #comment out if getting accuracy
      if test_set == 'SemEval':
        test = load_text.load_text(test_sets[test_set])
        steps_per_epoch = int(test._num_examples/batch_size)
      else:
        train, test = load_text.load_text(test_sets[test_set], 0.1)
        steps_per_epoch = int(test._num_examples/batch_size)
  
      #score = model.evaluate_generator(generator(test,max_len,batch_size),
      #    steps=steps_per_epoch)
      #print(score[1], end = ",")
      #score = conv_accuracy(test, model, max_len, max_len, steps_per_epoch)     
      
      scenario=model_name_short+"_"+str(max_len)+"_"+test_set
      
      scores = model.predict_generator(generator(test,max_len,batch_size),steps=steps_per_epoch)
      with open(scenario+"_default.csv", 'w') as out:
        for i in range(len(scores)):
          print(str(scores[i][0])+","+str(scores[i][1])+","+str(int(test.labels[i])), file=out)
      conv_output(test, model, max_len, max_len, steps_per_epoch, scenario) 
 

  #avoid randomly occuring error at end of code
  K.clear_session()
    
if __name__ == "__main__":
    main()
