# %%
'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
from numpy.lib.histograms import histogram
from generate_training_validation_data import TextMaskImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

# dimensions of our images.
img_width, img_height = 64, 64
train_data_dir = 'D:/MemeMachine_ProjectData/dataset/training'
validation_data_dir = 'D:/MemeMachine_ProjectData/dataset/validation'
epochs = 50
batch_size = 32

# %%
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=input_shape))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Dropout(0.5)) TODO consider applying dropout

# model.add(Dense(img_width*img_height))
# model.add(Activation('sigmoid'))

model = Sequential()
# model.add(Dense(img_width*img_height))
# model.add(Activation('relu'))
model.add(Dense(img_width*img_height))
model.add(Activation('sigmoid'))

model.compile(loss='MeanSquaredError', # Mean Squared Error should be good for this application because we are really just trying to predict whether or not a given pixel is in the mask
              optimizer='Adam',
              metrics=['accuracy'])

# %%
partition = dict()
partition['train'] = [] #list of image ids ("i.e., the file name of the image")
partition['validation'] = [] #list of image ids ("i.e., the file name of the image")

import os
#load the data
for file_ in  os.listdir(train_data_dir+"/"):
    partition['train'].append(str(train_data_dir+"/"+file_))
for file_ in  os.listdir(validation_data_dir+"/"):
    partition['validation'].append(str(validation_data_dir+"/"+file_))

nb_train_samples = 1 #len(partition['train'])
nb_validation_samples = 1 #len(partition['validation'])

# %%
# Generators
training_generator = TextMaskImageDataGenerator(partition['train'], x_size=img_width, y_size=img_height, batch_size=batch_size)
validation_generator = TextMaskImageDataGenerator(partition['validation'], x_size=img_width, y_size=img_height, batch_size=batch_size)

# %%
X, y = training_generator.__getitem__(0)
import cv2 as cv
# for i in range(X.shape[0]):
#     cv.imshow('X'+str(i),X[i, :, :, :])
for i in range(y.shape[0]):
    cv.imshow('y'+str(i),y[i, :, :, :])
# # print("X[0]", X[0])
# # print("X[0, :, :, :]", X[0, :, :, :])
cv.waitKey(0)




# %%
# model.fit(x=training_generator,
#     steps_per_epoch=1,
#     epochs=1,
#     validation_data=validation_generator,
#     validation_steps=1,
#     use_multiprocessing=True,
#     workers=11, verbose=1)

# %%
# model.fit(x=training_generator,
#     steps_per_epoch=nb_train_samples // batch_size,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=nb_validation_samples // batch_size,
#     use_multiprocessing=True,
#     workers=11, verbose=1)

# # %%
# model.save_weights('first_try.h5')


# # %%
# model.build((img_width, img_height, 3))
# model.summary()


