# This code is developed by CS Lab at Chosun University.
# Author: CS Lab team
# Date: 2023-11-11
# Version 1.0

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import glob
import cv2
import random
import np_utils
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.utils import to_categorical


# image pre-processing
def preprocess(image):
    # making the grayscale image
    grayscale_image = image.convert('L')

    # filtering the background
    image_array = np.array(grayscale_image)
    pivot_pixel_value = np.mean(image_array) * 0.8
    filtered_image_array = np.where(image_array < pivot_pixel_value, 0, image_array)
    filtered_image_array = np.where(filtered_image_array >= pivot_pixel_value, 255, filtered_image_array)
    filtered_image = Image.fromarray(filtered_image_array)

    # resize the input image
    image_size = 28
    resized_image = filtered_image.resize((image_size,image_size))

    return resized_image


# image opaque abstraction
def OA (image):
    # Get dimensions of the image
    height, width = image.size

    # convert to matrix
    img = np.array(image)

    # defining the abstraction vector
    n = (height + width) * 2
    opaque = np.ones(n)
    
    # threshold of opaque pixel
    opaque_threshold = 100

    # image padding
    padding = 5

    # TD scan
    for col in range(width): # loop on columns
        for row in range(padding,height): # scan each column top-down
            if img[row][col] < opaque_threshold :
                opaque[col] = row / height
                break

    # BU scan
    for col in range(width): # loop on columns
        for row in range(height-padding,-1,-1): # scan each column bottom-up
            if img[row][col] < opaque_threshold :
                opaque[width+col] = (height-row) / height
                break
            
    # RL scan
    for row in range(height): # loop on rows
        for col in range(width-padding,-1,-1): # scan each row right to left
            if img[row][col] < opaque_threshold :
                opaque[2*width+row] = (width-col) / width
                break

    # LR scan
    for row in range(height): # loop on rows
        for col in range(padding,width): # scan each row left to right
            if img[row][col] < opaque_threshold :
                opaque[2*width+height+row] = col / width
                break
    
    return opaque


# image glass abstraction
def GA (image):
    # Get dimensions of the image
    height, width = image.size

    # convert to matrix
    img = np.array(image)

    # defining the abstraction vector
    n = (height + width) * 2
    glass = np.zeros(n)
    
    # MD scan
    for col in range(width): # loop on columns
        shadow = 0
        for row in range(int(height/2)): # scan each column middle-down
            shadow += img[row][col]
        glass[col] = shadow / (height * 255)

    # MU scan
    for col in range(width): # loop on columns
        shadow = 0
        for row in range(int(height/2),height): # scan each column middle-up
            shadow += img[row][col]
        glass[width+col] = shadow / (height * 255)
            
    # ML scan
    for row in range(height): # loop on rows
        shadow = 0
        for col in range(int(width/2)): # scan each row middle to left
           shadow += img[row][col]
        glass[2*width+row] = shadow / (width * 255)

    # MR scan
    for row in range(height): # loop on rows
        shadow = 0
        for col in range(int(width/2),width): # scan each row middle to right
            shadow += img[row][col]
        glass[2*width+height+row] = shadow / (width * 255)
    
    return glass

# train and test the Neural Network
def ANN(x_train,y_train,x_test,y_test,abstraction_method):

    # one hot encoding of class labels
    y_train_ANN = to_categorical(y_train)
    y_test_ANN = to_categorical(y_test)

    tf.random.set_seed(4*28)

    # defining the ANN model
    model = keras.Sequential()

    # add input layer
    model.add(keras.Input(shape=np.array(x_train[1]).shape))

    # add hidden layer
    model.add(keras.layers.Dense(128, activation="relu"))

    if abstraction_method == "glass":
        # add second hidden layer
        model.add(keras.layers.Dense(128, activation="relu"))
    
    # add output layer
    model.add(keras.layers.Dense(len(set(y_train)), activation="softmax"))

    # print the model summary
    print("-"*40)
    print(model.summary())

    # setting model parameters
    model.compile(optimizer='adam', 
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    # train
    history = model.fit(x_train,y_train_ANN,epochs=50)
    
    # save trained model to a file
    model.save(f"{abstraction_method}.keras")

    # test
    y_pred = model.predict(x_test)

    # calculate the number of matches
    matched = 0
    for  i in range((y_pred.shape[0])):
        if np.argmax(y_pred[i]) == np.argmax(y_test_ANN[i]):
            matched += 1
    
    # calculate and print the model accuracy
    ACC = (matched / y_pred.shape[0]) * 100
    print("ACC:" , ACC)

    return y_pred


# -----------------------------------------------------------------------------------------------------#
#                                               Main module                                            #
# -----------------------------------------------------------------------------------------------------#

# image abstraction method
abstraction_method = "opaque" # opaque / glass
    
dataset = "data"
x_data = []
y_data = []
data_paths = []

print("-"*40)
print("start program ...")
print("abstraction method: ", abstraction_method)

# all class labels
labels = [name for name in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, name))]

# reading all input data paths
for i in range(len(labels)):
    folder_path = (glob.glob(f"{dataset}/{labels[i]}/*"))
    for file_name in folder_path:
        data_paths.append(file_name)
        y_data.append(labels[i])

# number of data items
N = len(data_paths)
#print("total data items: ", N)

print("-"*40)
print("start reading and abstraction of input data")

# preprocessing input images to remove the background
for path in data_paths:
    # read the input image
    input_image = Image.open(path)

    # pre-processing the input image
    processed_image = preprocess(input_image)

    # image abstraction
    if abstraction_method == "opaque":
        processed_image = OA(processed_image)
    elif abstraction_method == "glass":
        processed_image = GA(processed_image)
    else:
        print("Error: unknown abstraction method!")
        exit(0)

    # add the abstracted image to the set of data
    x_data.append(processed_image)

print("abstraction finished!")

print("-"*40)
print("start sampling for train and test ...")

# selecting 20% of data randomly as test set (selecting indices)
test_indices = random.sample(range(N), int(N/5))

#making a dictionary of class labels
dict_lable = {}
index = 0
for i in set(y_data):
    dict_lable[i] = index
    index += 1

# separating test set
x_test = []
y_test = []
for i in test_indices:
    x_test.append(x_data[i])
    y_test.append(dict_lable[y_data[i]])

# separating train set
x_train = []
y_train = []
for i in range(N):
    if i not in test_indices:
        x_train.append(x_data[i])
        y_train.append(dict_lable[y_data[i]])

# printing test, train, and data sizes
print("-"*40)
print("dataset size:", N)
print("test set size: ", len(x_test))
print("train set size: ", len(x_train))

print("-"*40)
print("start train and test the Neural Network!")

# convert list of numpy 3D arrays to a 4D array
x_train = np.stack(x_train, axis=0) 
x_test = np.stack(x_test, axis=0) 

# train and test ANN
a = ANN(x_train,y_train,x_test,y_test,abstraction_method)
