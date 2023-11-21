import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
from tensorflow.random import set_seed
from tensorflow import keras
from PIL import Image
import random

def Opaqu(x_train,x_test,image_size):
    train_len = len(x_train)
    test_len = len(x_test)
    Xp=np.ones([train_len,4*image_size])*image_size
    Xt=np.ones([test_len,4*image_size])*image_size
    
    for c in range(train_len):
        for i in range(np.array(x_train[c]).shape[0]):
            for j in range(5,np.array(x_train[c]).shape[1]):
                if np.array(x_train[c])[i][j] < 100 :
                    Xp[c][i] = j
                    break
                    
        for j in range(np.array(x_train[c]).shape[1]):
            for i in range(np.array(x_train[c]).shape[0]-1-5,-1,-1):
                if np.array(x_train[c])[i][j] < 100 :
                    Xp[c][image_size+j] = image_size-i
                    break
                    
        for i in range(np.array(x_train[c]).shape[0]):
            for j in range(np.array(x_train[c]).shape[1]-1-5,-1,-1):
                if np.array(x_train[c])[i][j] < 100:
                    Xp[c][2*image_size+i] = image_size-j
                    break

        for j in range(np.array(x_train[c]).shape[1]):
            for i in range(5,np.array(x_train[c]).shape[0]):
                if np.array(x_train[c])[i][j] < 100 :
                    Xp[c][3*image_size+j] = i
                    break

    for c in range(test_len):
        
        for i in range(np.array(x_test[c]).shape[0]):
            for j in range(5,np.array(x_test[c]).shape[1]):
                if np.array(x_test[c])[i][j]  < 100:
                    Xt[c][i]=j
                    break
                    
        for j in range(np.array(x_test[c]).shape[1]):
            for i in range(np.array(x_test[c]).shape[0]-1-5,-1,-1):
                if np.array(x_test[c])[i][j] < 100:
                    Xt[c][image_size+j] = image_size-i
                    break
                    
        for i in range(np.array(x_train[c]).shape[0]):
            for j in range(np.array(x_train[c]).shape[1]-1-5,-1,-1):
                if np.array(x_test[c])[i][j]  < 100:
                    Xt[c][2*image_size+i] = image_size-j
                    break

        for j in range(np.array(x_test[c]).shape[1]):
            for i in range(5,np.array(x_test[c]).shape[0]):
                if np.array(x_test[c])[i][j]  < 100:
                    Xt[c][3*image_size+j] = i
                    break
    for i in range(train_len):
        for k in range(image_size*4):
            Xp[i][k] = Xp[i][k]/image_size
    for i in range(test_len):
        for k in range(image_size*4):
            Xt[i][k] = Xt[i][k]/image_size
    return Xp , Xt


def glass(x_train,x_test,image_size):
    train_len = len(x_train)
    test_len = len(x_test)
    Xp=np.zeros([train_len,4*image_size])
    Xt=np.zeros([test_len,4*image_size])
    for i in range(train_len):
        for k in range(image_size):
            a=0
#             print(image_size/2)
            for h in range(int(image_size/2)):
                a=a+x_train[i][k][h]
            Xp[i][k]=a
        for k in range(image_size):
            a=0
            for h in range(int(image_size/2),image_size):
                a=a+x_train[i][k][h]
            Xp[i][image_size+k]=a
        for k in range(image_size):
            a=0
            for h in range(int(image_size/2)):
                a=a+x_train[i][h][k]
            Xp[i][2*image_size+k]=a
        for k in range(image_size):
            a=0
            for h in range(int(image_size/2),image_size):
                a=a+x_train[i][h][k]
            Xp[i][3*image_size+k]=a

    for i in range(test_len):
        for k in range(image_size):
            a=0
            for h in range(int(image_size/2)):
                a=a+x_test[i][k][h]
            Xt[i][k]=a
        for k in range(image_size):
            a=0
            for h in range(int(image_size/2),image_size):
                a=a+x_test[i][k][h]
            Xt[i][k+image_size]=a
        for k in range(image_size):
            a=0
            for h in range(int(image_size/2)):
                a=a+x_test[i][h][k]
            Xt[i][2*image_size+k]=a
        for k in range(image_size):
            a=0
            for h in range(int(image_size/2),image_size):
                a=a+x_test[i][h][k]
            Xt[i][3*image_size+k]=a
    for i in range(train_len):
        for k in range(image_size*4):
            Xp[i][k]=Xp[i][k]/(255*image_size)
    for i in range(test_len):
        for k in range(image_size*4):
            Xt[i][k]=Xt[i][k]/(255*image_size)
    return Xp , Xt

def finde_second(list_i):
    mx = max(list_i[0], list_i[1])
    secondmax = min(list_i[0], list_i[1])
    secondmax_index=0
    mx_index=0
    n = len(list_i)
    for i in range(2,n):
        if list_i[i] > mx:
            secondmax = mx
            mx = list_i[i]
        elif list_i[i] > secondmax and \
            mx != list_i[i]:
            secondmax = list_i[i]
        elif mx == secondmax and \
            secondmax != list_i[i]:
              secondmax = list_i[i]
    list_return=[]
    for i in range(n):
        if list_i[i]==max(list_i):
            mx_index=i
            list_return.append(max(list_i))
            list_return.append(mx_index)
    for i in range(n):
        if list_i[i]==secondmax:
            secondmax_index=i
            list_return.append(secondmax)
            list_return.append(secondmax_index)
    return list_return

def numbers_localmax(X):
    c=0
    for i in range(1,len(X)-1):
        if X[i]>X[i-1] and X[i]>=X[i+1]:
            c+=1
    return c

def ANN_Opaqu(y_train,Xp_Opaqu,Xt_Opaqu,y_test):
    y_train_ANN = np_utils.to_categorical(y_train)
    y_test_ANN = np_utils.to_categorical(y_test)
    set_seed(4*28)
    inputs = keras.Input(shape=Xp_Opaqu.shape[1])
    hidden_layer = keras.layers.Dense(128, activation="relu")(inputs)
    output_layer = keras.layers.Dense(35, activation="softmax")(hidden_layer)
    model = keras.Model(inputs=inputs, outputs=output_layer)
    print(model.summary())
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy())
    history = model.fit(Xp_Opaqu, y_train_ANN, epochs=50)
    model.save('model_Opaqu')
    y_pred = model.predict(Xt_Opaqu)
    cc=0
    for  i in range((y_pred.shape[0])):
        if np.argmax(y_pred[i]) == np.argmax(y_test_ANN[i]):
            cc+=1
    ACC=((cc/y_pred.shape[0])*100)
    print(ACC)
    return y_pred

def ANN_len(y_train,Xp_len,y_test,Xt_len):
    y_train_ANN = np_utils.to_categorical(y_train)
    y_test_ANN = np_utils.to_categorical(y_test)
    set_seed(4*28)
    inputs = keras.Input(shape=Xp_len.shape[1])
    hidden_layer = keras.layers.Dense(128, activation="relu")(inputs)
    hidden_layer_1 = keras.layers.Dense(128, activation="relu")(hidden_layer)
    output_layer = keras.layers.Dense(35, activation="softmax")(hidden_layer_1)
    model = keras.Model(inputs=inputs, outputs=output_layer)
    print(model.summary())
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy())
    history = model.fit(Xp_len, y_train_ANN, epochs=50)
#     sns.lineplot(x=history.epoch, y=history.history['loss'])
    model.save('model_glass')
    y_pred = model.predict(Xt_len)
    print(y_pred.shape)
    cc=0
    for  i in range((y_pred.shape[0])):
        if np.argmax(y_pred[i]) == np.argmax(y_test_ANN[i]):
            cc+=1
    ACC=((cc/y_pred.shape[0])*100)
    print(ACC)
    print("**")
    return ACC


art=0

for i in range(len(train)):
    b=(glob.glob(f"Path to data folder/{train[i]}/*"))   
    art=art+(len(b))

x_train_path=[]
y_train=[]
for i in range(len(train)):
    b=(glob.glob(f"Path to data folder/{train[i]}/*"))
    for j in b:
#         print(b)
        x_train_path.append(j)
        y_train.append(train[i])

x_train=[]
for i in x_train_path:
    input_image = Image.open(i)
    new_size = (28, 28)
    resized_image = filtered_image.resize(new_size)
    grayscale_image = resized_image.convert('L')
    image_array = np.array(grayscale_image)
    pivot_pixel_value = np.mean(image_array) * 0.8
    filtered_image_array = np.where(image_array < pivot_pixel_value, 0, image_array)
    filtered_image_array = np.where(filtered_image_array >= pivot_pixel_value, 255, filtered_image_array)
    filtered_image = Image.fromarray(filtered_image_array)
    x_train.append(filtered_image)

test_index=random.sample(range(0, 42000-1), 4200)

x_test=[]
y_test=[]
for i in test_index:
    print(i)
    x_test.append(x_train[i])
    y_test.append(y_train[i])
x_train_final=[]
y_train_final=[]
for i in range(42000):
    if i not in test_index:
        x_train_final.append(x_train[i])
        y_train_final.append(y_train[i])

x_train_Opaqu , x_test_Opaqu =  Opaqu(x_train_final,x_test,28)  
x_train_len , x_test_len =  glass(x_train,x_test,200)
dict_lable={}
index=0
for i in set(y_test):
    dict_lable[i]=index
    index+=1

for i in range(len(y_train_final)):
    y_train_final[i]=dict_lable[y_train_final[i]]
for i in range(len(y_test)):
    y_test[i]=dict_lable[y_test[i]]

a=ANN_Opaqu(y_train_final,x_train_Opaqu,x_test_Opaqu,y_test) 