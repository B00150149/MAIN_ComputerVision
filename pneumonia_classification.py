

from __future__ import print_function

import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Rescaling, BatchNormalization, GlobalAveragePooling2D
from keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast
from keras.optimizers import RMSprop,Adam
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import keras_tuner as kt
# from tensorflow.keras.applications import VGG16

batch_size = 12
num_classes = 3
epochs = 8
img_width = 128
img_height = 128
img_channels = 3
fit = True #make fit false if you do not want to train the network again
train_dir = 'C:\\Users\\yashi\\Documents\\01_tud_yashika\\4th_year\Computer Vision\\chest_xray\\train'
test_dir = 'C:\\Users\\yashi\\Documents\\01_tud_yashika\\4th_year\Computer Vision\\chest_xray\\test'

with tf.device('/gpu:0'):
    
    #create training,validation and test datatsets
    train_ds,val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        seed=123,
        validation_split=0.2,
        subset='both',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        shuffle=True)
    
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        seed=None,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        shuffle=True)

    class_names = train_ds.class_names
    print('Class Names: ',class_names)

    #Q2: Class Weight Distributin
    def count_images(folder):
        counts = {}
        for cls in os.listdir(folder):
            counts[cls] = len(os.listdir(os.path.join(folder, cls)))
        return counts

    train_counts = count_images(train_dir)
    test_counts = count_images(test_dir)

    print("Train distribution:", train_counts)
    print("Test distribution:", test_counts)

    num_classes = len(class_names)
    
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(2):
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i].numpy()])
            plt.axis("off")
    plt.show()

    labels = []

    for _, y in train_ds.unbatch():
        labels.append(y.numpy())

    labels = np.array(labels)
    counts = np.bincount(labels)
    weights = np.max(counts)/counts
    class_weight = {cls: weight for cls, weight in enumerate(weights)}

    print("Counts:", counts)
    print("Class weights:", class_weight)

    #Q5 create model FUNCTION USUSING KERAS TUNER
    def build_model():
        #Q6) Transfer Learning Approach
        pretrained_model = tf.keras.applications.VGG16(
            include_top = False,
            input_shape=(img_height,img_width, img_channels),
            weights='imagenet')
        
        #Load pretrained model and freeze the weights so they are not modofies during the training
        for layer in pretrained_model.layers[:-2]:
            layer.trainable=False


        model = tf.keras.Sequential([

            #Q4 DATA AUGMENTATAION
            RandomFlip("horizontal"),
            RandomRotation(0.1),
            RandomZoom(0.1),
            # RandomContrast(0.1),

            Rescaling(1.0/255),

            #add the pretrainedmodel
            pretrained_model,

            #Q3: Fixing Overfitting 
            GlobalAveragePooling2D(),
            Dense(256, activation='relu'),
            Dropout(0.4),         
            Dense(num_classes, activation = 'softmax')
        ])

        model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=Adam(learning_rate=1e-4),
                    metrics=['accuracy'])
    

        return model 
       
    model = build_model()
    start_time = time.time()
        
    history = model.fit(
        train_ds,
        class_weight = class_weight,
        validation_data=val_ds,
        epochs=epochs
    )
        
    end_time = time.time()
    elapsed = end_time - start_time
    print(f'\nTraining time: {elapsed:.1f}s  ({elapsed/60:.1f} minutes)')
 
    score = model.evaluate(test_ds) #',batch_size=batch_size
    print('Test accuracy:', score[1])

    
    #if fit:
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    test_batch = test_ds.take(1)
    plt.figure(figsize=(10, 10))
    for images, labels in test_batch:
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            prediction = model.predict(tf.expand_dims(images[i].numpy(),0))#perform a prediction on this image
            plt.title('Actual:' + class_names[labels[i].numpy()]+ '\nPredicted:{} {:.2f}%'.format(class_names[np.argmax(prediction)], 100 * np.max(prediction)))
            plt.axis("off")
    plt.show()
