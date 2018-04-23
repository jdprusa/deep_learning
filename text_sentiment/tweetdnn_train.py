# code for training character-level DNN in keras with log(m) text embedding

from __future__ import print_function

import keras as keras
from keras import callbacks
from models import load_model
from keras import backend as K

import fau_models
import load_text
import resnet

# hyperparameters
batch_size = 200
num_classes = 2
epochs = 27
val_split = 0.1
max_len = 750
input_shape = (8,max_len,1)

# file directory dependant
train_data = ""
base_dir = ""
#model_dir = ""

# custom batch generator that peforms character embedding within each batch for memory efficiency
def generator(data, batch_size):
  while True:
    text, labels_, labels, chunks  = data.next_batch(batch_size, max_len, max_len, 'chop')
    yield text, keras.utils.to_categorical(labels, num_classes)

def main():

  # load data
  train, val = load_text.load_text(train_data, val_split)
  steps_per_epoch = int(train._num_examples/batch_size)
  validation_steps = int(val._num_examples/batch_size)
  
  #build model, comment if loading existing model
  model = fau_models.text_cnn_big(input_shape, num_classes)
  #model = resnet.ResnetBuilder.build_resnet_152(input_shape, num_classes, text=True)
  
  model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False),
              metrics=['accuracy'])
              
  #load model, uncomment if loading existing model
  #model = load_model(model_dir)
  
  #callbacks
  cb  = []
  cb.append(callbacks.ModelCheckpoint(filepath=base_dir+str(max_len)+'/model.{epoch:02d}-{val_loss:.4f}.h5', monitor='val_loss', save_best_only=True, mode='min'))
  cb.append(callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001))
  cb.append(callbacks.CSVLogger(base_dir+str(max_len)+'/epoch_results.csv', separator=',', append=True))
  
  #fit model
  model.fit_generator(generator(train,batch_size), 
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1,
        validation_data=generator(val,batch_size),
        validation_steps=validation_steps,
        callbacks=cb)
        
  K.clear_session()
    
if __name__ == "__main__":
    main()
