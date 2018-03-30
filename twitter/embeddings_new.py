import csv
import numpy as np
np.random.seed(7)  # for reproducibility
import string
import argparse

def text2image(text, max_len):
   #print "Converting to image format 8x longest tweet"
    image = np.zeros((8, max_len), dtype=np.int8)
    #print text
    for index, character in enumerate(text):
        #print index
        if index < max_len:
            y = list(bin(ord(character)))
            if len(y) <=10:
                for j in range(2,min(8,len(y))):
                    image[j-2, index] = int(y[len(y)+1-j])
    #print image.shape
    return image
    
def text2image_old(text):
   #print "Converting to image format 8x longest tweet"
    image = np.zeros((8, len(text)), dtype=np.int8)
    #print text
    for index, character in enumerate(text):
        #print index
        y = list(bin(ord(character)))
        if len(y) <=10:
            for j in range(2,min(8,len(y))):
                image[j-2, index] = int(y[len(y)+1-j])
    #print image.shape
    return image
    
def load_dataset(data_file, max_len=50, stride=50, chop_text='chop'):
    print("Loading Dataset")
    #max_len = 0
    data =[]
    with open(data_file) as file:
        f = csv.reader(file)
        print("Converting to image format")
        #count = 0
        if chop_text == 'full_pad':
            print("padded to maximum instance length embedding")
            for line in f:
                instance = text2image_old(line[1])
                length = instance.shape[1]
                if length > max_len:
                    max_len = length
                data.append((instance, line[0]))
        elif chop_text == 'conv':
            print("convolving window embedding")
            for line in f:
                for i in range(0,len(line[1])+stride-max_len, stride):
                    instance = text2image(line[1][i:i+max_len], max_len)
                    data.append((instance, line[0]))
        else:
            print("chopped by max_len embedding")
            for line in f:
                instance = text2image(line[1], max_len)
                data.append((instance, line[0]))
            
    np.random.shuffle(data)
 
    X = []
    Y = []
    for instance in data:
        X.append(instance[0])
        if instance[1] == '4':
            Y.append(np.int8(1))
        else:
            Y.append(np.int8(0))
     
    if chop_text == 'full_pad':
        print("Padding Data")
        for i, entry in enumerate(X):     
            npad = ((0, 0), (0, max_len-entry.shape[1]))
            X[i] = np.pad(entry, pad_width=npad, mode='constant', constant_values=0)

        print("Padding complete")
    X = np.asarray(X)
    Y = np.asarray(Y)
    print("Reshaping array")
    
    X = X.reshape(-1, 8, max_len, 1)
    print(X.shape)
    return X, Y, max_len
  
def batch_embedding(batch_data, batch_labels, max_len, stride, chop_text):
    data =[]
    chunks=1
    if chop_text == 'full_pad':
        labels = batch_labels
        for text in batch_data:
            instance = text2image_old(text)
            length = instance.shape[1]
            if length > max_len:
                max_len = length
            data.append(instance)
    elif chop_text == 'conv':
        labels = []
        chunks = []
        chunk_count=0
        for i in range(len(batch_data)):
            for j in range(0,max((len(batch_data[i])+stride-max_len),1), stride):
                chunk_count+=1
                instance = text2image(batch_data[i][j:j+max_len], max_len)
                data.append(instance)
                labels.append(batch_labels[i])
            chunks.append(chunk_count)
            chunk_check = {}

              

            
    else:
        labels = batch_labels
        for text in batch_data:
            instance = text2image(text, max_len)
            data.append(instance)
 
    if chop_text == 'full_pad':
        for i, entry in enumerate(data):     
            npad = ((0, 0), (0, max_len-entry.shape[1]))
            data[i] = np.pad(entry, pad_width=npad, mode='constant', constant_values=0)

    data = np.asarray(data)
    data = data.reshape(-1, 8, max_len, 1)
    
    return data, labels, chunks

def output_to_file(path, x, y):
    print("Writing out output file")
    with open(path, 'w') as outfile:
        for (instance, label) in zip(x, y):
            outfile.write(instance)
            outfile.write(',')
            outfile.write(label)
            outfile.write('\n')

#test of functions to check if np.arrays are the correct dimensions
def main():  
    X, Y, _ = load_dataset(data_file = FLAGS.input_data_dir, max_len=FLAGS.max_len, stride=FLAGS.conv_stride, chop_text=FLAGS.chop_text)
    print(X[0].shape, X.shape, Y[0], Y.shape)
    print(X[0])
    #X=text2image("test input", FLAGS.max_len)
    #print(X)
    #print(X.shape)
    #X = X.reshape(-1, 8, FLAGS.max_len, 1)
    #print(X.shape)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--max_len',
      type=int,
      default=50,
      help='max number of characters'
    )
    
    parser.add_argument(
      '--embedding',
      type=str,
      default='2D',
      help='character embedding'
    )
    
    parser.add_argument(
      '--input_data_dir',
      type=str,
      default='sentiment140_100k.csv',
      help='Directory to put the input data.'
    )
    
    parser.add_argument(
      '--chop_text',
      type=str,
      default='chop',
      help='determines if characters over max_len are chopped, or if a window of length max_len is convolved across the instance'
    )
    
    parser.add_argument(
      '--conv_stride',
      type=int,
      default=50,
      help='determines stride if a window of length max_len is convolved across the instance'
    )
    
    FLAGS, unparsed = parser.parse_known_args()
    main()
    
    
