

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
from sklearn.metrics import classification_report
from tf_explain.core.grad_cam import GradCAM

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

    # ------------------------------------------------------------------------------
    #Q2: Class Weight Distributin
    # ------------------------------------------------------------------------------
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

    # ------------------------------------------------------------------------------
    #Q2 CLASS WEIGHTS after
    # ------------------------------------------------------------------------------
    labels = []

    for _, y in train_ds.unbatch():
        labels.append(y.numpy())

    labels = np.array(labels)
    counts = np.bincount(labels)
    weights = np.max(counts)/counts
    #Finding the VIRAL index
    viral_idx = class_names.index('VIRAL')
    #Q8 )Incraese VIRAL by a factor for balance in classification
    boost_factor = 2.0
    weights[viral_idx] *= boost_factor
    class_weight = {cls: weight for cls, weight in enumerate(weights)}

    print("Counts:", counts)
    print("Class weights:", class_weight)

    # ------------------------------------------------------------------------------ 
    #Q5 create model FUNCTION USUSING KERAS TUNER
    # ------------------------------------------------------------------------------

    def build_model():


        inputs = tf.keras.Input(shape=(img_height, img_width, img_channels))

        # ------------------------------------------------------------------------------
        #Q4 DATA AUGMENTATAION
        # ------------------------------------------------------------------------------
        x = RandomFlip("horizontal")(inputs)
        x = RandomRotation(0.2)(x)
        x = RandomZoom(0.2)(x)
        x = RandomContrast(0.2)(x)

        x = Rescaling(1.0/255)(x)

        # ------------------------------------------------------------------------------
        #Q6) Transfer Learning Approach
        # ------------------------------------------------------------------------------
        base_model  = tf.keras.applications.VGG16(
            include_top = False,
            weights='imagenet',
            input_tensor=x)
        
        base_model.trainable = True

        #Load pretrained model and freeze the weights so they are not modofies during the training
        for layer in base_model .layers[:-4]:
            layer.trainable=False


         # ------------------------------------------------------------------------------
        #Q3: Fixing Overfitting 
        # ------------------------------------------------------------------------------
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.4)(x)
        outputs = Dense(num_classes, activation='softmax')(x)


        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=Adam(learning_rate= 1e-4),
                    metrics=['accuracy'])
    
        return model
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)
    save_callback = tf.keras.callbacks.ModelCheckpoint("pneumonia.keras",save_freq='epoch',save_best_only=True)
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6) 


    model = build_model()
    start_time = time.time()
        
        
    history = model.fit(
        train_ds,
        class_weight = class_weight,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[earlystop_callback, save_callback, lr_callback]
    )
        
    end_time = time.time()
    elapsed = end_time - start_time
    print(f'\nTraining time: {elapsed:.1f}s  ({elapsed/60:.1f} minutes)')


    #if shuffle=True when creating the dataset, samples will be chosen randomly   
    score = model.evaluate(test_ds) #, batch_size=batch_size
    print('Test accuracy:', score[1])

    # ------------------------------------------------------------------------------
    #Q7) The per class precision, recall and F1 scores
    # ------------------------------------------------------------------------------
    #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    y_true, y_pred =[], []
    for images, labels in test_ds:
        preds = model.predict(images)
        y_pred.extend(np.argmax(preds, axis=1))  #converted probabities to class indices
        y_true.extend(labels.numpy())

    print(classification_report(y_true, y_pred, target_names=class_names))

    # ------------------------------------------------------------------------------
    # Q9) tf-explain (e.g.GradCam) to see what the CNN is seeing
    # ------------------------------------------------------------------------------
    # https://deepwiki.com/sicara/tf-explain/4.1.1-gradcam
    print("\nRunning Grad-CAM")

    explainer = GradCAM()

    for images, labels in test_ds.take(1):

        image = images[0].numpy()
        label = labels[0].numpy()

        prediction = model.predict(np.expand_dims(image, axis=0))
        pred_class = np.argmax(prediction)

        #LOOP THROUGH ALL CLASSES (THIS IS THE KEY CHANGE)
        for i, class_name in enumerate(class_names):

            data = (np.expand_dims(image, axis=0), None)

            grid = explainer.explain(
                data,
                model,
                class_index=i, 
                layer_name="block5_conv3"
            )

            # CLEAN VISUALIZATION
            plt.figure(figsize=(5,5))
            plt.imshow(image.astype("uint8"))
            plt.imshow(grid, cmap="jet", alpha=0.4)
            plt.title(f"Grad-CAM for {class_name}")
            plt.axis("off")
            plt.show()

        break


    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
        
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
