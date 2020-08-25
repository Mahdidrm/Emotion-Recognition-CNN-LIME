# Emotion-Recognition
In this work, we followed two steps: 
1- We used a convolution neural network (CNN) for the emotions recognition
2- Using the Lime algorithm to monitor the heat maps of the input image which help our CNN decide to select the corresponding emotion.

Main Code
-
# Import the needed libraries
- from __future__ import print_function
- import keras
- from keras.preprocessing.image import ImageDataGenerator
- from keras.models import Sequential
- from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
- from keras.layers import Conv2D, MaxPooling2D
- from keras.preprocessing.image import ImageDataGenerator
- import os
- from keras.models import Sequential
- from keras.layers.normalization import BatchNormalization
- from keras.layers.convolutional import Conv2D, MaxPooling2D
- from keras.layers.advanced_activations import ELU
- from keras.layers.core import Activation, Flatten, Dropout, Dense
- from keras.optimizers import RMSprop, SGD, Adam
- from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
- from keras import regularizers
- from keras.regularizers import l1

# Initialing data
- num_classes = 7                                        # We have 7 Classes: Angry, Disgust, Fear, Happy, Natural, Sad and Surprise
- img_rows, img_cols = 48,48                             # The size of input image
- batch_size = 512                                       # The number of input pixels on model
- train_data_dir = '/src/fer2013/train'                  # Train data(contain 7 subfolder for each emotion)
- validation_data_dir = '/src/fer2013/validation/'       # Test data(contain 7 subfolder for each emotion)

Use some data augmentaiton 
-
#val_datagen = ImageDataGenerator(rescale=1./255)
#train_datagen = ImageDataGenerator(
#rescale=1./255,
#rotation_range=30,
#shear_range=0.3,
#zoom_range=0.3,
#horizontal_flip=True,
#fill_mode='nearest')
#train_generator = train_datagen.flow_from_directory(
#train_data_dir,
#target_size=(48,48),#(48,48),
#batch_size=batch_size,
#color_mode="grayscale",
#class_mode='categorical')

#validation_generator = val_datagen.flow_from_directory(
#validation_data_dir,
#target_size=(48,48), #(48,48),
#batch_size=batch_size,
#color_mode="grayscale",
#class_mode='categorical')
        
Creating the model
-
# For the first step, we used a handcrafted architecture from CNN with these layers: 
- model = Sequential()

- model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',kernel_regularizer=regularizers.l2(0.0001),input_shape=(48,48,1)))
- model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
- model.add(MaxPooling2D(pool_size=(2, 2)))

- model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
- model.add(MaxPooling2D(pool_size=(2, 2)))

- model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
- model.add(MaxPooling2D(pool_size=(2, 2)))

- model.add(Conv2D(6, kernel_size=(1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
- model.add(Conv2D(7, kernel_size=(4, 4), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))

- model.add(Flatten())
- model.add(Activation("softmax"))
- model.summary()

# About Layers:
- Each conv2D layer extracts the features of the image according to the kernel filter. we used 3 * 3 kernels.
- And on the Maxpool diapers; they select the entities with the highest resolutions among the 2 * 2 dimension entities.
- Flatten Layer converts a tensor to a vector to send it to fully connected layers that use in the classification.
- The softmax activation is normally applied to the very last layer in a neural net, instead of using ReLU, sigmoid, tanh, or another activation function. The reason why softmax is useful is because it converts the output of the last layer in your neural network into what is essentially a probability distribution.


