import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

########33Data preprocessing#####
train_data_gen=ImageDataGenerator(rescale=1.0/255,zoom_range=0.2,shear_range=0.2,horizontal_flip=True)
training_set=train_data_gen.flow_from_directory(r"C:\Users\prans\OneDrive\Desktop\DL_Project\CNN_Data\training_set",batch_size=32,target_size=(64,64),class_mode="binary")

test_data_gen=ImageDataGenerator(rescale=1.0/255)
test_set=test_data_gen.flow_from_directory(r"C:\Users\prans\OneDrive\Desktop\DL_Project\CNN_Data\test_set",batch_size=32,target_size=(64,64),class_mode="binary")

########Building CNN##############

cnn=tf.keras.models.Sequential()
## Adding first CL
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))
##Max Pool Layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=None))

## Adding another CL-
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=None))

#Flattening to create full connection and apply neural network
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
##Output layer
cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid')) #No of classes -1 =2-1=1

##
cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
cnn.fit(x=training_set,validation_data=test_set,epochs=20) #batch_size is default

import numpy as np
from keras.preprocessing import image as ip
test_image=ip.load_img(r"C:\Users\prans\OneDrive\Desktop\DL_Project\CNN_Data\test_set\cat_test.jpg",target_size=(64,64))
test_image=ip.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=cnn.predict(test_image/255.0)
print(result)
print(training_set.class_indices)
if result[0][0]>0.50:
    prediction="dog"
else:
    prediction="cat"
print(prediction)




