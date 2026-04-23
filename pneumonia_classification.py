

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
epochs = 20
img_height = 224
img_width = 224
img_channels = 3

train_dir = 'C:\\Users\\yashi\\Documents\\01_tud_yashika\\4th_year\\Computer Vision\\chest_xray\\train'
test_dir = 'C:\\Users\\yashi\\Documents\\01_tud_yashika\\4th_year\\Computer Vision\\chest_xray\\test'

with tf.device('/gpu:0'):


    #create training,validation and test datatsets
    train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset='both',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical'
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical'
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print('Class Names: ', class_names)

    # ------------------------------------------------------------------------------
    #Q2 CLASS WEIGHTS after
    # ------------------------------------------------------------------------------
    labels = []
    for _, y in train_ds.unbatch():
        labels.append(np.argmax(y.numpy()))

    labels = np.array(labels)
    counts = np.bincount(labels)
    weights = np.max(counts) / counts
    class_weight = {i: w for i, w in enumerate(weights)}

    bacterial_idx = class_names.index('BACTERIAL')
    viral_idx = class_names.index('VIRAL')

    class_weight[bacterial_idx] *= 1.8  #1.5
    class_weight[viral_idx] *= 1.5
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
        x = tf.keras.layers.RandomFlip("horizontal")(inputs)
        x = tf.keras.layers.RandomRotation(0.1)(x)
        x = tf.keras.layers.RandomContrast(0.2)(x)
        x = tf.keras.layers.RandomTranslation(0.1, 0.1)(x)
        x = tf.keras.layers.RandomZoom(0.1)(x)

        # Preprocessing
        x = tf.keras.applications.efficientnet.preprocess_input(x)

        # ------------------------------------------------------------------------------
        #Q6) Transfer Learning Approach
        # ------------------------------------------------------------------------------
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_tensor=x
        )

        # Freeze base
        for layer in base_model.layers:
            layer.trainable = False

        # ------------------------------------------------------------------------------
        #Q3: Fixing Overfitting 
        # ------------------------------------------------------------------------------
        # Head
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

        model = tf.keras.Model(inputs, outputs)

        loss = tf.keras.losses.CategoricalFocalCrossentropy(
            gamma=1.5,
            alpha=0.4
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
            loss=loss,
            metrics=['accuracy']
        )

        return model, base_model


    model, base_model = build_model()
    model.summary()

    # -----------------------
    # CALLBACKS
    # -----------------------
    earlystop_callback = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3)
    save_callback = tf.keras.callbacks.ModelCheckpoint("pneumonia.keras", save_best_only=True)

    # -----------------------
    # TRAIN (basic phase 1)
    # -----------------------
    start_time = time.time()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=[earlystop_callback, reduce_lr, save_callback]
    )

    # -----------------------
    # FINE-TUNING
    # -----------------------
    for layer in base_model.layers[-100:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-5),
        loss=tf.keras.losses.CategoricalFocalCrossentropy(gamma=1.5, alpha=0.25),
        metrics=['accuracy']
    )

    print("\nFine-tuning...")

    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10, #10
        class_weight=class_weight,
        callbacks=[earlystop_callback, reduce_lr]
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
    y_true, y_pred = [], []

    for images, labels in test_ds:
        preds = model.predict(images)
        y_pred.extend(np.argmax(preds, axis=1))   #converted probabities to class indices
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    print(classification_report(y_true, y_pred, target_names=class_names))

    # ------------------------------------------------------------------------------
    # Q9) tf-explain (e.g.GradCam) to see what the CNN is seeing
    # ------------------------------------------------------------------------------
    # https://deepwiki.com/sicara/tf-explain/4.1.1-gradcam
    print("\nRunning Grad-CAM...")
    explainer = GradCAM()

    for images, labels in test_ds.take(1):

        image = images[0].numpy()
        label = labels[0].numpy()

        prediction = model.predict(np.expand_dims(image, axis=0))
        pred_class = np.argmax(prediction)

        #LOOP THROUGH ALL CLASSES
        for i, class_name in enumerate(class_names):

            data = (np.expand_dims(image, axis=0), None)

            grid = explainer.explain(
                data,
                model,
                class_index=i, 
                layer_name="top_activation"
            )

            # CLEAN VISUALIZATION
            plt.figure(figsize=(5,5))
            plt.imshow(image.astype("uint8"))
            plt.imshow(grid, cmap="jet", alpha=0.4)
            plt.title(f"Grad-CAM for {class_name}")
            plt.axis("off")
            plt.show()


    #Plot Graph +Accuracy
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']

    plt.figure()
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'Val'], loc='upper left')
    plt.show()

    test_batch = test_ds.take(1)
    plt.figure(figsize=(10, 10))
    for images, labels in test_batch:
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            img = images[i].numpy().astype("uint8")
            plt.imshow(img)
            prediction = model.predict(tf.expand_dims(img, 0), verbose=0)
            true_label = np.argmax(labels[i].numpy())
            pred_label = np.argmax(prediction)
            plt.title(f"Actual: {class_names[true_label]}\n" f"Predicted: {class_names[pred_label]} ({100 * np.max(prediction):.2f}%)")
            plt.axis("off")
    plt.show()