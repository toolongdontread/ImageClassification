import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # error msg type
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau

####################################################################################################
# (most of the) parameters and hyperparameters 

if __name__ == '__main__':

    IMG_SIZE = 112
    BATCH_SIZE = 8
    EPOCH = 25 # 20 is ard stability
    LR = 0.0003

    train_dataset_path = './data/train'
    validation_dataset_path = './data/val'
    #test_path will move so not put here 

####################################################################################################
# data preparation & augmentation

    train_datagen = ImageDataGenerator(horizontal_flip=True,
                                       zca_whitening=True,
                                       rescale=1/255,
                                       brightness_range=(0.8,1),
                                       channel_shift_range=60,
                                       featurewise_std_normalization=True)
    train_generator = train_datagen.flow_from_directory(train_dataset_path,
                                                        target_size=(IMG_SIZE, IMG_SIZE),
                                                        batch_size=BATCH_SIZE,
                                                        color_mode="rgb",
                                                        shuffle=True)

    validation_datagen = ImageDataGenerator(horizontal_flip=True,
                                            zca_whitening=True,
                                            brightness_range=(0.8,1),
                                            channel_shift_range=60,
                                            featurewise_std_normalization=True,
                                            rescale=1/255)
    validation_generator = validation_datagen.flow_from_directory(validation_dataset_path,
                                                                  target_size=(IMG_SIZE, IMG_SIZE),
                                                                  batch_size=BATCH_SIZE,
                                                                  color_mode="rgb",
                                                                  shuffle=False)
    
    # train_generator & validation_generator is a 5-d array
    # array structure as follows:

    # - idx[0..220 (No. of BATCH-1)] : Batch No.
    #   - idx[0] : Image                             - idx[1] : Classes             
    #     - idx[0..7 (BATCH_SIZE-1)] : Elem. ID        - idx[0..7 (BATCH_SIZE-1)] : Elem. ID
    #       - idx[0..111 (IMG_SIZE-1)] : Row No.         - idx[0..19(No. of class-1)] : Class No.
    #         - idx[0..2] : Red/Green/Blue                 - value[0..1] : One-hot code, Not-belongs/Belongs
    #           - value[0..255] : RGB value

 #################################################################################################### 
 # sample image display (for better understanding of data preprocessing)
 
    """
    labels = {value: key for key, value in train_generator.class_indices.items()}
    print("Label Mappings for classes present in the training and validation datasets\n")
    # for function to run, ROW x COL must be <= BATCH_SIZE
    ROW = 2
    COL = 4

    for key, value in labels.items():
        print(f"{key} : {value}")

    fig, ax = plt.subplots(nrows=ROW, ncols=COL, figsize=(10, 4))
    idx = 0
    
    for i in range(ROW):
        for j in range(COL):
            label = labels[np.argmax(train_generator[0][1][idx])]
            ax[i, j].set_title(f"{label}")
            ax[i, j].imshow(train_generator[0][0][idx][:, :, :])
            ax[i, j].axis("off")
            idx += 1
    
    plt.tight_layout()
    plt.suptitle("Sample Training Images", fontsize=20)
    plt.show()
    """
    
####################################################################################################
# model building
    cnn_model = Sequential([
        Conv2D(filters=64, kernel_size=(5), padding='valid', input_shape=(IMG_SIZE, IMG_SIZE, 3), activation='relu'), # change 3 according to color-mode
        MaxPooling2D(pool_size=(3, 3), padding='valid'),
        BatchNormalization(),
        
        Conv2D(filters=256, kernel_size=(3), padding='valid', kernel_regularizer=l2(0.00005), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
    
        Conv2D(filters=256, kernel_size=(3), padding='valid', kernel_regularizer=l2(0.00005), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(filters=256, kernel_size=(3), padding='valid', kernel_regularizer=l2(0.00005), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        
        Flatten(),
        Dense(units=128,activation='relu'),
        Dense(units=128, activation='sigmoid'),
        Dense(units=20, activation='softmax')
    ])
    
    """
    # for better understanding of model building
    print(cnn_model.summary())
    """
    
####################################################################################################
# model implementation

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5)
    optimizer = Adam(learning_rate=LR)
    cnn_model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(), metrics=['accuracy'])
    history = cnn_model.fit(train_generator, 
                            epochs=EPOCH, 
                            validation_data=validation_generator,
                            verbose=1,
                            callbacks=[reduce_lr])
    
####################################################################################################
# data presentation (final result)
    #"""
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    learning_rate = history.history['lr']
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 10))

    ax[0].set_title('Training Accuracy vs. Epochs')
    ax[0].plot(train_accuracy, 'o-', label='Train Accuracy')
    ax[0].plot(val_accuracy, 'o-', label='Validation Accuracy')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend(loc='best')

    ax[1].set_title('Training/Validation Loss vs. Epochs')
    ax[1].plot(train_loss, 'o-', label='Train Loss')
    ax[1].plot(val_loss, 'o-', label='Validation Loss')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].legend(loc='best')

    ax[2].set_title('Learning Rate vs. Epochs')
    ax[2].plot(learning_rate, 'o-', label='Learning Rate')
    ax[2].set_xlabel('Epochs')
    ax[2].set_ylabel('Loss')
    ax[2].legend(loc='best')

    plt.tight_layout()
    plt.show()
    #"""
    cnn_model.save('./trained_model2.h5')
    print('model saved!')
    