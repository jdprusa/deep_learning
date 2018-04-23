# Dataset class for deep learning with Keras with a Tensorflow Backend 
# generates train, test and validation splits, and training batches    

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy
import embeddings_new
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed


#DataSet class for general use with any data
class DataSet(object):

    def __init__(self, data, labels):      
        self._num_examples = len(data)
        self._data = data
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def features(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed
    # Return the next `batch_size` examples from this data set.
    def next_batch(self, batch_size, max_len, stride, chop_text, shuffle=True):
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:			
            c = list(zip(self._data, self._labels))
            numpy.random.shuffle(c)
            self._data, self._labels = zip(*c)
            # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start			#grabs remaining data if batch > num_exampls
            data_rest_part = self._data[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:								
                c = list(zip(self._data, self._labels))
                numpy.random.shuffle(c)
                self._data, self._labels = zip(*c)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples		#restarts at the beginning getting whatever amount of data left to reach batch quantity
            end = self._index_in_epoch
            data_new_part = self._data[start:end]
            labels_new_part = self._labels[start:end]
            text_data, labels = numpy.concatenate((data_rest_part, data_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)		#adds remaining data to end data and returns
            text_data, labels_, chunks = embeddings_new.batch_embedding(text_data, labels, max_len, stride, chop_text)
            return text_data, numpy.asarray(labels_, dtype=numpy.int8), numpy.asarray(labels, dtype=numpy.int8), chunks
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            text_data, labels = self._data[start:end], self._labels[start:end]
            text_data, labels_, chunks = embeddings_new.batch_embedding(text_data, labels, max_len, stride, chop_text)
            return text_data, numpy.asarray(labels_, dtype=numpy.int8), numpy.asarray(labels, dtype=numpy.int8), chunks
            
 
# generic dataloader for DataSet Class
# requires data, labels as nparrays
# validation_size and test_size determine how much of the data will be set aside for validation and testing
def read_data_sets(data, labels,
                   validation_split=0.2,
                   test_split=0.1):

    #calculate train, validation, test partition locations
    if validation_split != 0:
        validation_size = int(len(labels)*validation_split)
        test_size = int(len(labels)*test_split)
    
        train_data = data[validation_size+test_size:]
        train_labels = labels[validation_size+test_size:]

        validation_data = data[test_size:validation_size+test_size]
        validation_labels = labels[test_size:validation_size+test_size]
    
        test_data = data[:test_size]
        test_labels = labels[:test_size]
    
    else:
        test_size = int(test_split*len(labels))
        
        train_data = data[test_size:]
        train_labels = labels[test_size:]

        validation_data = numpy.zeros(5)
        validation_labels = numpy.zeros(5)
    
        test_data = data[:test_size]
        test_labels = labels[:test_size]
        
    #converts sections of the datasets into the DataSet objects
    train = DataSet(train_data, train_labels)
    validation = DataSet(validation_data, validation_labels)
    test = DataSet(test_data, test_labels)
    


    return base.Datasets(train=train, validation=validation, test=test)
