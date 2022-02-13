import pandas as pd
import numpy as np

#loading the csv data
data=pd.read_csv(r"C:\Users\prans\OneDrive\Desktop\DL_Project\P16-Deep-Learning-AZ\Deep_Learning_A_Z\Volume 1 - Supervised Deep Learning\Part 1 - Artificial Neural Networks (ANN)\Section 4 - Building an ANN\Part 1 - Artificial Neural Networks\Churn_Modelling.csv")
#/r is used to consider this path string as raw
#print(data.head())

x=data.iloc[:,3:13] #row,col
# print(x.head(10)) #print starting 10 records
# print(x.columns) #print names of columns   , ctrl+/ for multiline comments
#print(x)

y=data.iloc[:,-1].values # :,13:14 or :,-1 for last column, .values convert in to numpy
#print(y)

one_hot_encoded_data = pd.get_dummies(x, columns=['Geography', 'Gender'])
x=one_hot_encoded_data.values #convert dataframe into nested list
#print(one_hot_encoded_data)



from sklearn.model_selection import train_test_split

# print(len(x))
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.70)
print(len(x_train))
print(len(x_test))
print(len(y_train))

#z = (x - u) / s
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn.preprocessing import StandardScaler #for scaling the data
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
print(x_train)

import tensorflow as tf
#from tensorflow import keras
classifier=tf.keras.models.Sequential()

#from keras.models import Sequential  #for building ANN
#from keras.layers.core import Dense
#from tensorflow.keras.layers import Dense
#adding input layer to first hidden layer
classifier.add(tf.keras.layers.Dense(units=6,activation='relu')) #(11+1)/2
#second hidden layer
classifier.add(tf.keras.layers.Dense(units=6,activation='relu'))
classifier.add(tf.keras.layers.Dense(units=6,activation='relu'))

#adding the output layer
classifier.add(tf.keras.layers.Dense(units=1,activation='sigmoid')) #sigmoid for probability

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])  #adam is algo for batch gradient descent
#adam is for back propagation comes under schotiant gradient
classifier.fit(x_train,y_train,batch_size=32,epochs=100)  #batch gradient descent

y_pred=classifier.predict(x_test)
print(y_pred)
y_pred=(y_pred>0.5)
print(y_pred)

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
ascore=accuracy_score(y_test,y_pred)
print(ascore)