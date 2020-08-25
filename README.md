# Emotion-Recognition
In this work, we followed two steps: 
1- We used a convolution neural network (CNN) for the emotions recognition
2- Using the Lime algorithm to monitor the heat maps of the input image which help our CNN decide to select the corresponding emotion.

For the first step, we used a handcrafted architecture from CNN with these layers:

#%% Creating the model
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
---------------------------------------------------
Each conv2D layer extracts the features of the image according to the kernel filter. we used 3 * 3 kernels.
And on the Maxpool diapers; they select the entities with the highest resolutions among the 2 * 2 dimension entities.
