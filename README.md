# Emotion-Recognition

### In this work, we followed two steps: 
```
1- We used a convolution neural network (CNN) for the emotions recognition
2- Using the Lime algorithm to monitor the heat maps of the input image which help our CNN decide to select the corresponding emotion.
```
### Requirements

- Python:               ```              3.3+ or Python 2.7           ```
- OS:                   ```              Windows macOS or Linux       ```
- Keras:                ```              pip install Keras            ```
- Tensorflow:           ```              pip install Tensorflow       ```
- dlib:                 ```              pip install dlib             ```
- face-recognition:     ```              pip install face-recognition ```


Main Code
-
```
Import the needed libraries
from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import regularizers
from keras.regularizers import l1
```
## Initialing data
```
num_classes = 7                                        # We have 7 Classes: Angry, Disgust, Fear, Happy, Natural, Sad and Surprise
img_rows, img_cols = 48,48                             # The size of input image
batch_size = 512                                       # The number of input pixels for augmentation
train_data_dir = '/src/fer2013/train'                  # Train data(contain 7 subfolder for each emotion)
validation_data_dir = '/src/fer2013/validation/'       # Test data(contain 7 subfolder for each emotion)
```
Use some data augmentation
-
- Use ImageDataGenerator to create fake samples to help our network learn better and avoid overfitting. We perform a rescale, rotation_range, shear_range, zoom_range, horizontal_flip to do training, validation and test data.
        
Creating the model
-
-   For the first step, we used a handcrafted architecture from CNN with these layers: 
```
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',kernel_regularizer=regularizers.l2(0.0001),input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(6, kernel_size=(1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(Conv2D(7, kernel_size=(4, 4), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))

model.add(Flatten())
model.add(Activation("softmax"))
model.summary()
```
### About Layers:
- Each conv2D layer extracts the features of the image according to the kernel filter. we used 3 * 3 kernels.
- And on the Maxpool diapers; they select the entities with the highest resolutions among the 2 * 2 dimension entities.
- Flatten Layer converts a tensor to a vector to send it to fully connected layers that use in the classification.
- The softmax activation is normally applied to the very last layer in a neural net, instead of using ReLU, sigmoid, tanh, or another activation function. The reason why softmax is useful is because it converts the output of the last layer in your neural network into what is essentially a probability distribution.

Training the model
-
In this step, using our augmented data, we start to train our model. 
```
 #First we need to load a model file to save the training results (model weight).
filepath = os.path.join('/emotion_detector_models/model.hdf5')  
#We simply monitor the true values of the validation data during training and record the best values.       
checkpoint = keras.callbacks.ModelCheckpoint(filepath,           
                                            monitor='val_acc',      
                                            verbose=1,
                                            save_best_only=True,
                                            mode='max')
callbacks = [checkpoint]

```
# Model compilation
At first we need to compile your model. We use Adam's optimization and cross entropy to reduce the loss value of our model.
```
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])  
Adam is an optimization algorithm that can be used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data.
nb_train_samples = 31205# 28709          #Number of train samples
nb_validation_samples = 6085 # 3589      #Number of test sample
epochs = 50                              #Number of train and test loob
```
# Training the model
The main line to train our model. We train our model to augmented training and validation data.
```
model_info = model.fit_generator(                  
                                train_generator,
                                steps_per_epoch=nb_train_samples // batch_size,
                                epochs=epochs,
                                callbacks = callbacks,
                                validation_data=validation_generator,
                                validation_steps=nb_validation_samples // batch_size)

model.save_weights('/emotion_detector_models/model.hdf5')
```
