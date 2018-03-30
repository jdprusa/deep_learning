import  keras
from keras.models import (
    Sequential
)
from keras.layers import (
    Dense,    
    Dropout, 
    Flatten 
)
from keras.layers.convolutional import (
    Conv2D, 
    MaxPooling2D,
)


def text_cnn_big(input_shape, num_classes):
    """
    simple deep NN for text classification using character level embedding
    2 blocks of 3 Conv layers followed by 2x2 maxpooling, then 3 fully connected layers
    """
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,
                 padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model
    
