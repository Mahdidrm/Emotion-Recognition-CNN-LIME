## Emotion-Recognition via CNN and Explaination AI (LIME algorithm)
The main idea of this work is to apply several XAI methods like LIME on the CNN classifier applying on emotion datasets. Fer2013 and CK + datasets are used via the proposed method and the result is presented in the following sections. As a novelty of our work,in the last part of this code, the LIME and LRP explanation AI algorithms are the objectives, but other methods will also be studied. 

### In this work, we followed two steps: 
 
1) We used a convolution neural network (CNN) for the emotions recognition 
2) Using the Lime algorithm to monitor the heat maps of the input image which help our CNN decide to select the corresponding emotion.


### Requirements

- Python:-------------------->             ```              3.3+ or Python 2.7                         ```
- OS:------------------------>             ```              Windows macOS or Linux                     ```
- Keras:--------------------->             ```              pip install Keras                          ```
- Tensorflow:--------------->             ```              pip install Tensorflow                     ```
- dlib:----------------------->             ```              pip install dlib                           ```
- face-recognition:---------->             ```              pip install face-recognition               ```
- LIME explaination AI:----->             ```              pip install lime                           ```
- pandas: ------------------>             ```             pip install pandas                         ```
- numpy:-------------------->             ```              pip install numpy                          ```
- h5py:---------------------->             ```              pip install h5py                           ```
- opencv-python---------->                ```             pip install opencv-python                   ```

- fer2013 dataset:----------->             ```              https://www.kaggle.com/deadskull7/fer2013  ```
- CK+ dataset: ----------->         ```       https://www.pitt.edu/~emotion/ck-spread.htm       ```

Fer2013 dataset
-
   The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

   train.csv contains two columns, "emotion" and "pixels". The "emotion" column contains a numeric code ranging from 0 to 6, inclusive, for the emotion that is present in the image. The "pixels" column contains a string surrounded in quotes for each image. The contents of this string a space-separated pixel values in row major order. test.csv contains only the "pixels" column and your task is to predict the emotion column.

   The training set consists of 28,709 examples. The public test set used for the leaderboard consists of 3,589 examples. The final test set, which was used to determine the winner of the competition, consists of another 3,589 examples.

   This dataset was prepared by Pierre-Luc Carrier and Aaron Courville, as part of an ongoing research project. They have graciously provided the workshop organizers with a preliminary version of their dataset to use for this contest.
   
### Note:
In our work we converted CSV files to the images.

CK+ dataset
- 
   In 2000, the Cohn-Kanade (CK) database was released for the purpose of promoting research into automatically detecting individual facial expressions. Since then, the CK database has become one of the most widely used test-beds for algorithm development and evaluation. During this period, three limitations have become apparent: 
   
 1) While AU codes are well validated, emotion labels are not, as they refer to what was requested rather than what was actually performed,
 
 2) The lack of a common performance metric against which to evaluate new algorithms, and 3) Standard protocols for common databases have not emerged.
 
 As a consequence, the CK database has been used for both AU and emotion detection (even though labels for the latter have not been validated), comparison with benchmark algorithms is missing, and use of random subsets of the original database makes meta-analyses difficult. To address these and other concerns, we present the Extended Cohn-Kanade (CK+) database. The number of sequences is increased by 22% and the number of subjects by 27%. The target expression for each sequence is fully FACS coded and emotion labels have been revised and validated. In addition to this, non-posed sequences for several types of smiles and their associated metadata have been added. We present baseline results using Active Appearance Models (AAMs) and a linear support vector machine (SVM) classifier using a leave-one-out subject cross-validation for both AU and emotion detection for the posed data. The emotion and AU labels, along with the extended image data and tracked landmarks will be made available July 2010.
 

## Main Code
Import the needed libraries
```
from __future__ import print_function
import tensorflow as tf
import tensorflow.keras
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
from tensorflow.python.client import device_lib
from keras import backend as K
import cv2
import lime
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb

```
Initialing data
- 
```
#%% Initioaling
num_classes =7                                 # We have 7 Classes: Angry, Disgust, Fear, Happy, Natural, Sad and Surprise
img_rows, img_cols = 48,48                     # The size of input images
batch_size = 50                               # The number of input pixels for augmentation

train_data_dir = '/CKFace481/training'         # Train data(contain 7 subfolder for each emotion)
validation_data_dir = '/CKFace481/validation'  # Test data(contain 7 subfolder for each emotion)
```
Use some data augmentation
-
- Use ImageDataGenerator to create fake samples to help our network learn better and avoid overfitting. We perform a rescale, rotation_range, shear_range, zoom_range, horizontal_flip to do training, validation and test data.
```
val_datagen = ImageDataGenerator(rescale=1./255)
train_datagen = ImageDataGenerator(
        rescale=1./255,
      rotation_range=30,
      shear_range=0.3,
      zoom_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="rgb",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(48,48), 
        batch_size=batch_size,
        color_mode="rgb",
        class_mode='categorical')
print(validation_generator.class_indices)

  ```   
Creating the model
-
-   For the first step, we used a handcrafted architecture from CNN with these layers: 
```
# Create the model

model =Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',padding='same', kernel_regularizer=regularizers.l2(0.0001),input_shape=(48,48,3)))
# model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',padding='same', kernel_regularizer=regularizers.l2(0.0001)))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',padding='same', kernel_regularizer=regularizers.l2(0.0001)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',padding='same', kernel_regularizer=regularizers.l2(0.0001)))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(6, kernel_size=(1, 1), activation='relu',padding='same', kernel_regularizer=regularizers.l2(0.0001)))
# model.add(BatchNormalization())
model.add(Conv2D(7, kernel_size=(4, 4), activation='relu',padding='same', kernel_regularizer=regularizers.l2(0.0001)))
# model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dense(7))
model.add(Activation("softmax"))
model.summary()
```
About Layers:
- 

- Each conv2D layer extracts the features of the image according to the kernel filter. we used 3 * 3 kernels.
- And on the Maxpool diapers; they select the entities with the highest resolutions among the 2 * 2 dimension entities.
- Flatten Layer converts a tensor to a vector to send it to fully connected layers that use in the classification.
- The softmax activation is normally applied to the very last layer in a neural net, instead of using ReLU, sigmoid, 
tanh, or another activation function. The reason why softmax is useful is because it converts the output of
the last layer in your neural network into what is essentially a probability distribution.

- How define the layer's dimensions

To create convolutional layer, we use tf.nn.conv2d. It computes a 2-D convolution given 4-D input and filter tensors.

- Inputs:

tensor of shape [batch, in_height, in_width, in_channels]. x of shape [batch_size,28 ,28, 1]
a filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]. W is of size [5, 5, 1, 32]
stride which is [1, 1, 1, 1]. The convolutional layer, slides the "kernel window" across the input tensor. As the input tensor has 4 dimensions: [batch, height, width, channels], then the convolution operates on a 2D window on the height and width dimensions. strides determines how much the window shifts by in each of the dimensions. As the first and last dimensions are related to batch and channels, we set the stride to 1. But for second and third dimension, we could set other values, e.g. [1, 2, 2, 1]

- Process:

Change the filter to a 2-D matrix with shape [5*5*1,32]
Extracts image patches from the input tensor to form a virtual tensor of shape [batch, 28, 28, 5*5*1].
For each batch, right-multiplies the filter matrix and the image vector.

- Output:

A Tensor (a 2-D convolution) of size tf.Tensor 'add_7:0' shape=(?, 28, 28, 32)- Notice: the output of the first convolution layer is 32 [28x28] images. Here 32 is considered as volume/depth of the output image

### Train the model

In this step, using our augmented data, we start to train our model. First we need to load a model file to save the training results (model weight):
```
filepath = os.path.join('/emotion_detector_models/model.hdf5')  
```
Then, we simply monitor the true values of the validation data during training and record the best values:
```
#%% Make some check points
import keras
checkpoint = keras.callbacks.ModelCheckpoint(filepath,
                                             monitor='val_acc',
                                             verbose=1,
                                             save_best_only=True,
                                             mode='max')
callbacks = [checkpoint]

```
### Model compilation
- At first we need to compile our model. We use Adam's optimization and cross entropy to reduce the loss value of our model.
```
#%%  Compile the model

model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
nb_train_samples = 2032
nb_validation_samples = 1235 
```
- Adam is an optimization algorithm that can be used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data.
```
nb_train_samples = 31205           #Number of train samples
nb_validation_samples = 6085       #Number of test sample
epochs = 50                        #Number of train and test loob
```
### Training Step

The main line to train our model. We train our model to augmented training and validation data.
```
#%% Training the model

epochs = 100
model_info = model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            callbacks = callbacks,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size)
```
Plot model loss in train step
-
```
print(model_info.history.keys())
from matplotlib import pyplot as plt
plt.plot(model_info.history['loss'])
plt.plot(model_info.history['val_loss'])
plt.title('Model loss in Train step')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```


- The loss of training step on CK+:
<p align="center">
<img align="center" width="400" height="400" src="https://github.com/Mahdidrm/Emotion-Recognition-CNN-Fer2013-Lime/blob/master/Figures/ck_train_loss.png?raw=true">
</p>

```
#%% Plot Model loss in validation step
from matplotlib import pyplot as plt

plt.plot(model_info.history['acc'])
plt.plot(model_info.history['val_acc'])
plt.title('Model accuracy rate in Test step')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```
- The accuracy of training step on CK+:

<p align="center">
<img align="center" width="400" height="400" src="https://github.com/Mahdidrm/Emotion-Recognition-CNN-Fer2013-Lime/blob/master/Figures/ck_train_acc.png?raw=true">
</p>

Confusion Matrix of our model in some validation images
-
- First, we need to import some libraries
```
#%% classification_report
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import sklearn.metrics as metrics
```
- Confution Matrix and Classification Report
```
test_generator = ImageDataGenerator()
test_data_generator = test_generator.flow_from_directory(
        validation_data_dir,
        color_mode = 'rgb',
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
test_steps_per_epoch = np.math.ceil(test_data_generator.samples / test_data_generator.batch_size)

predictions = model.predict_generator(test_data_generator, steps=test_steps_per_epoch)
# Get most likely class
predicted_classes = np.argmax(predictions, axis=1)


true_classes = test_data_generator.classes
class_labels = list(test_data_generator.class_indices.keys())   

report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
cmatrix=metrics.confusion_matrix(y_true=validation_generator.classes, y_pred=predicted_classes)

print(report) 
print(cmatrix)
```
<p align="center">
<img align="center" width="400" height="400" src="https://github.com/Mahdidrm/Emotion-Recognition-CNN-LIME/blob/master/Figures/report.png?raw=true">
</p>    
- Confusion Matrix in graphic mode: 
```
#%% Confusion matrix as graphic - not complete
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

cmatrix = confusion_matrix(y_true=validation_generator.classes, y_pred=predicted_classes)
classes=['Angry','Disgust','Fear','Happy','Natural','Sad','Surprise']
thresh = cmatrix.max() / 2.
# print(cmatrix)
# print(cm)

plt.imshow(cmatrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('true_classes')
plt.xlabel('predicted_classes')
for i, j in itertools.product(range(cmatrix.shape[0]), range(cmatrix.shape[1])):
        plt.text(j, i, cmatrix[i, j],
            horizontalalignment="center",
            color="white" if cmatrix[i, j] > thresh else "black")
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.title('Confusion matrix')
plt.colorbar()
```
- And output is: 
<p align="center">
<img align="center" width="400" height="400" src="https://github.com/Mahdidrm/Emotion-Recognition-CNN-Fer2013-Lime/blob/master/Figures/matrix.png?raw=true">
</p>           
Explainable AI (XAI) 
-
Explainable AI (XAI) refers to methods and techniques in the application of artificial intelligence technology (AI) such that the results of the solution can be understood by humans. It contrasts with the concept of the "black box" in machine learning where even their designers cannot explain why the AI arrived at a specific decision.[1] XAI may be an implementation of the social right to explanation.[2] XAI is relevant even if there is no legal rights or regulatory requirements (for example, XAI can improve the user experience of a product or service by helping end users trust that the AI is making good decisions.)

The technical challenge of explaining AI decisions is sometimes known as the interpretability problem.[3] Another consideration is info-besity (overload of information), thus, full transparency may not be always possible or even required. However, simplification at the cost of misleading users in order to increase trust or to hide undesirable attributes of the system should be avoided by allowing a tradeoff between interpretability and completeness of an explanation. [4]

AI systems optimize behavior to satisfy a mathematically-specified goal system chosen by the system designers, such as the command "maximize accuracy of assessing how positive film reviews are in the test dataset". The AI may learn useful general rules from the test-set, such as "reviews containing the word 'horrible'" are likely to be negative". However, it may also learn inappropriate rules, such as "reviews containing 'Daniel Day-Lewis' are usually positive"; such rules may be undesirable if they are deemed likely to fail to generalize outside the test set, or if people consider the rule to be "cheating" or "unfair". A human can audit rules in an XAI to get an idea how likely the system is to generalize to future real-world data outside the test-set.[3]


XAI Methods
-
- LIME

LIME is a novel explanation technique that explains the predictions of any classifier in an interpretable and faithful manner, by learning an interpretable model locally around the prediction. [5]

In lime we have two main steps:
<p align="center">
<img align="center" width="700" height="500" src="https://github.com/Mahdidrm/Emotion-Recognition/blob/master/Figures/12.jpg?raw=true">
</p> 
- “Preprocessing”

1) The ground truth image is first segmented into different sections using Quickshift segmentation.
2) The next step is to generate N data samples by randomly masking out some of the image regions based on the segmentations.
This is resulted in a data matrix of "samples x segmentations" where the first row is kept with no mask applied (all 1).
3) Each sample is weighted based on how much it is different from the original vector (row 1) using some ‘distance’ function.

- “Explanation”
4) Each data sample (masked/pertubed image) is passed to the classifier (the model being explained e.g. our emotion classifier) for the prediction.
5) The data instances (binaries) with the corresponding weights (step 3) and the predicted label (step 4- but one label at the time) are then fit to the K-LASSO or Ridge regression classifier to measure the importance of each feature (segmentation in this case).
- Ridge Regression is a technique for analyzing multiple regression data that suffer from multicollinearity. When multicollinearity occurs, least squares estimates are unbiased, but their variances are large so they may be far from the true value.
6) The final output are the weights, which are representing significance of each segmented feature on the given class.
7) The positive (support) and negative (against) segments/features are display based on the given thresholding value
(e.g. ‘0’ as the separating boundary of being supportive or not).

- LIME code for our work:
```
#Import the libraries
from keras.models import load_model
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
```
- At firs we need to load our saved model
```
model.load_weights(os.path.join('/modelCK48-cnn1.hdf5'))
```
- Now we define our fuctions:
1) new_predict_fn uses to reshape the input image, then calculate its prediction.

2) new_predict_fn_proba uses to find the probability class of the input image.

3) getLabel uses to find the class tag corresponding to the problematic class in the image. This means that when we get the probability of an image, we need to get the class name of that number to display in our plot.

```
def new_predict_fn(img):
    img = np.asarray(img, dtype = np.float32)
    # normalizing the image
    img = img / 255
    # reshaping the image into a 4D array
    img = img.reshape( -1, 48, 48, 3)  
    return model.predict(img)

def new_predict_fn_proba(img):
    img = np.asarray(img, dtype = np.float32)
    # normalizing the image
    img = img / 255
    # reshaping the image into a 4D array
    img = img.reshape( -1, 48, 48, 3)  
    return model.predict_proba(img)

def getLabel(id):
    return ['Angry', 'Disgust','Fear','Happy','Natural', 'Sad','Surprise'][id]

def new_predict_generator_fn(img):
    img = np.asarray(img, dtype = np.float32)
    # normalizing the image
    img = img / 255
    # reshaping the image into a 4D array
    img = img.reshape( -1, 48, 48, 3)  
    return model.predict_generator(img,validation_generator, nb_validation_samples // batch_size+1)

- And we put a Mask on image to show the HeatMaps
```
- Now, according to LIME algorithm, we should define our explainer and our segmentation algorithm:
```
explainer = lime_image.LimeImageExplainer(verbose = False)
segmenter = SegmentationAlgorithm('slic', n_segments=100, compactness=1, sigma=1)
```
- We load an image as our input:
```
img5 = cv2.imread("U:/Emotion/Classifications/NEW-CNN/00/1/9481_4.jpg")
```
- In this step, with the help of two explained functions "new_predict_fn" and "new_predict_fn_proba" we obtain the prediction and the probability of the input image:
```
pipe_pred_test = new_predict_fn(img5)
pipe_pred_prop = new_predict_fn_proba(img5)
```
- And now we do our explanation using the input image, the prediction of the classifier model and other parameters.
```
explanation=explainer.explain_instance(img5, 
                                      classifier_fn = model.predict_proba, 
                                      top_labels=7, hide_color=0, num_samples=10000, segmentation_fn=segmenter)
```
- At the end of the algorithm, we trace the results. Below we have checked the likelihood of image belonging to each class.
```
fig, m_axs = plt.subplots(1,7, figsize = (15,7))
for i, (c_ax) in zip(explanation.top_labels, m_axs.T):
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                positive_only=True, 
                                                num_features=5,
                                                hide_rest=False,
                                                min_weight=0.01)
    c_ax.imshow(label2rgb(mask,temp, bg_label = 0), interpolation = 'nearest')
    c_ax.set_title('Positive for {}\nScore:{:2.2f}%'.format(getLabel(i), 100*pipe_pred_prop[0, i]))
    c_ax.axis('off')

 ```
- Outputs:

1) Angry input image of CK+ dataset:
![](https://github.com/Mahdidrm/Emotion-Recognition-CNN-Fer2013-Lime/blob/master/Figures/CNN1-learn-via-cK/Lime-ck-angry.png?raw=true)
2) Disgust input image of CK+ dataset:
![](https://github.com/Mahdidrm/Emotion-Recognition-CNN-Fer2013-Lime/blob/master/Figures/CNN1-learn-via-cK/Lime-ck-disgust.png?raw=true)
3) Fear input image of CK+ dataset:
![](https://github.com/Mahdidrm/Emotion-Recognition-CNN-Fer2013-Lime/blob/master/Figures/CNN1-learn-via-cK/Lime-ck-fear.png?raw=true)
4) Happy input image of CK+ dataset:
![](https://github.com/Mahdidrm/Emotion-Recognition-CNN-Fer2013-Lime/blob/master/Figures/CNN1-learn-via-cK/Lime-ck-happy.png?raw=true)
5) Natural input image of CK+ dataset:
![](https://github.com/Mahdidrm/Emotion-Recognition-CNN-Fer2013-Lime/blob/master/Figures/CNN1-learn-via-cK/Lime-ck-natural.png?raw=true)
6) Sad input image of CK+ dataset:
![](https://github.com/Mahdidrm/Emotion-Recognition-CNN-Fer2013-Lime/blob/master/Figures/CNN1-learn-via-cK/Lime-ck-sad.png?raw=true)
7) Surprise input image of CK+ dataset:
![](https://github.com/Mahdidrm/Emotion-Recognition-CNN-Fer2013-Lime/blob/master/Figures/CNN1-learn-via-cK/Lime-ck-surprise.png?raw=true)

  
Refrences
-
[1] Sample, Ian (5 November 2017). "Computer says no: why making AIs fair, accountable and transparent is crucial". the Guardian. Retrieved 30 January 2018. https://www.theguardian.com/science/2017/nov/05/computer-says-no-why-making-ais-fair-accountable-and-transparent-is-crucial
 
[2] Edwards, Lilian; Veale, Michael (2017). "Slave to the Algorithm? Why a 'Right to an Explanation' Is Probably Not the Remedy You Are Looking For". Duke Law and Technology Review. 16: 18. SSRN 2972855. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2972855

[3] "How AI detectives are cracking open the black box of deep learning". Science. 5 July 2017. Retrieved 30 January 2018. https://www.sciencemag.org/news/2017/07/how-ai-detectives-are-cracking-open-black-box-deep-learning
 
[4] Gilpin, Leilani H.; Bau, David; Yuan, Ben Z.; Bajwa, Ayesha; Specter, Michael; Kagal, Lalana (2018-05-31). "Explaining Explanations: An Overview of Interpretability of Machine Learning". arXiv:1806.00069 [stat.AI]. https://arxiv.org/abs/1806.00069

[5] Paper: https://arxiv.org/abs/1602.04938  Code: https://github.com/marcotcr/lime 
