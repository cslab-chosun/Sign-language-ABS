import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import pickle
from skimage import color
from data_pre import datagenerator
from tensorflow import keras
import seaborn as sns
from numpy.random import seed
from tensorflow.random import set_seed
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import picklez
from data_pre import datagenerator
from keras.datasets import mnist
from keras.utils import np_utils
from train_ANN import ANN_chart
from train_ANN import ANN_len
from train_SVM import SVM_len
from train_SVM import SVM_chart
from extra_keras_datasets import emnist

def chart(x_train,x_test,image_size):
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


def len_(x_train,x_test,image_size):
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

def ANN_chart(y_train,Xp_chart,Xt_chart,y_test):
    y_train_ANN = np_utils.to_categorical(y_train)
    y_test_ANN = np_utils.to_categorical(y_test)
#     model=keras.models.load_model('model_Ann_letter_chart_mnist_')
    set_seed(4*28)
    inputs = keras.Input(shape=Xp_chart.shape[1])
    hidden_layer = keras.layers.Dense(128, activation="relu")(inputs)
    output_layer = keras.layers.Dense(35, activation="softmax")(hidden_layer)
    model = keras.Model(inputs=inputs, outputs=output_layer)
    print(model.summary())
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy())
    history = model.fit(Xp_chart, y_train_ANN, epochs=50)
#     sns.lineplot(x=history.epoch, y=history.history['loss'])
    model.save('model_Ann_chart_finger_45')
    y_pred = model.predict(Xt_chart)
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
#     model=keras.models.load_model('model_Ann-letter_len_mnist_')
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
    model.save('model_Ann_len_finger')
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
    grayscale_image = input_image.convert('L')
    image_array = np.array(grayscale_image)
    pivot_pixel_value = np.mean(image_array) * 0.8
    filtered_image_array = np.where(image_array < pivot_pixel_value, 0, image_array)
    filtered_image_array = np.where(filtered_image_array >= pivot_pixel_value, 255, filtered_image_array)
    filtered_image = Image.fromarray(filtered_image_array)
    new_size = (28, 28)
    resized_image = filtered_image.resize(new_size)
    x_train.append(resized_image)

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

x_train_chart , x_test_chart =  chart(x_train_final,x_test,28)  
x_train_len , x_test_len =  len_(x_train,x_test,200)
dict_lable={}
index=0
for i in set(y_test):
    dict_lable[i]=index
    index+=1

for i in range(len(y_train_final)):
    y_train_final[i]=dict_lable[y_train_final[i]]
for i in range(len(y_test)):
    y_test[i]=dict_lable[y_test[i]]

a=ANN_chart(y_train_final,x_train_chart,x_test_chart,y_test) 