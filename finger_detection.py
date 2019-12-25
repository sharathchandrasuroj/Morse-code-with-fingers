"@Author: @UdayKiran"
#Check Jupyter notebook file for dataset visualization

#importing Packages
import numpy as np 
import pandas as pd 
from PIL import Image
import os, glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import zipfile

#Data set used https://www.kaggle.com/koryakinp/fingers
#File structure of dataset
#I'm using Image Data Generator so the dataset files should be in this format
"""
Fingers
|
|
|---------Train
|         |
|         |-------0
|         |      |1.jpg
|         |      |2.jpg
|         |      | .
|         |      | .
|         |-------1
|         |      |1.jpg
|         |      |2.jpg
|         |      | .
|         |      | .
|         |-------2
|         |      |1.jpg
|         |      |2.jpg
|         |      | .
|         |      | .
|         |-------3
|         |      |1.jpg
|         |      |2.jpg
|         |      | .
|         |      | .
|         |-------4
|         |      |1.jpg
|         |      |2.jpg
|         |      | .
|         |      | .
|         |-------5
|         |      |1.jpg
|         |      |2.jpg
|         |      | .
|         |      | .
|---------Test
|         |
|         |-------0
|         |      |1.jpg
|         |      |2.jpg
|         |      | .
|         |      | .
|         |-------1
|         |      |1.jpg
|         |      |2.jpg
|         |      | .
|         |      | .
|         |-------2
|         |      |1.jpg
|         |      |2.jpg
|         |      | .
|         |      | .
|         |-------3
|         |      |1.jpg
|         |      |2.jpg
|         |      | .
|         |      | .
|         |-------4
|         |      |1.jpg
|         |      |2.jpg
|         |      | .
|         |      | .
|         |-------5
|         |      |1.jpg
|         |      |2.jpg
|         |      | .
|         |      | .
"""
#Fingers is the Root folder and inside that we have two folders called Train and Test.
#Inside each folder we have 0,1,2,3,4,5 as subfolders which consists of respective finger images.
#0,1,2,3,4,5 are class lables for our dataset

#0 - no fingers opened
#1 - one finger opened
#2 - two fingers opened
#3 - three fingers opened
#4 - four fingers opened
#5 - five fingers opened

Class FingerModel():

    def __init__():
        pass

    #initializing the image generator of test and train
    def imageGenerator(self, train_path, test_path):
        #you can tweak all the parameters i'm giving the best parameters list which i found"
        batch = 128 #Batch size of image generaor 

        train_datagenerator = ImageDataGenerator( rescale=1./255,
                                            rotation_range=10.,
                                            width_shift_range=0.1,
                                            height_shift_range=0.1,
                                            zoom_range=0.2,
                                            horizontal_flip=True
                                          )

        test_datagenerator  = ImageDataGenerator( rescale=1./255 )

        training = train_datagenerator.flow_from_directory(
                train_path,
                target_size=(150, 150),
                color_mode='grayscale',
                batch_size=batch,
                classes=['0','1','2','3','4','5'],
                class_mode='categorical'
            )

        testing = test_datagenerator.flow_from_directory(
                test_path,
                target_size=(150, 150),
                color_mode='grayscale',
                batch_size=batch,
                classes=['0','1','2','3','4','5'],
                class_mode='categorical'
            )

        return training, testing

    def model_architecture(self):
        #you can tweak all the parameters i'm giving the best parameters list which i found"
        """
           Here i'm using 2D Conv because we converted our dataset to gray scale images of size 150X150
        """
        model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 1)),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(6, activation='softmax')
                ])
        print(model.summary())
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        return model

    def train_model(self, train_data, test_data, model):
        #you can tweak all the parameters i'm giving the best parameters list which i found"
        no_of_epochs = 10
        no_of_steps_per_epoch = 140
        no_of_validation_steps = 30
        
        history = model.fit_generator(
                  train_data,
                  steps_per_epoch = no_of_steps_per_epoch,  
                  epochs = no_of_epochs, 
                  validation_data = test_data,
                  validation_steps = no_of_validation_steps)

        model.save("my_model.h5")#saving the model for future use.

    def main(self):
        train_data_path =  "fingers/train/"
        test_data_path =  "fingers/test/"
        train_datagen, test_datagen = self.imageGenerator(train_data_path, test_data_path)
        model = self.model_architecture()
        self.train_model(train_datagen, test_datagen, model)
        return "success"


if __name__:"__main__":
    main()

        
    

