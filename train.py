import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import cv2
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau

# train

if __name__ == '__main__':
    
    print(tf.__version__)


    train_dataset_path = './DATA/train'
    validation_dataset_path = './DATA/val'

    IMG_WIDTH = 112
    IMG_HEIGHT = 112
    BATCH_SIZE = 32

    train_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow_from_directory(train_dataset_path,
                                                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    shuffle=True)

    validation_datagen = ImageDataGenerator()
    validation_generator = validation_datagen.flow_from_directory(validation_dataset_path,
                                                                target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                                batch_size=BATCH_SIZE,
                                                                class_mode='categorical',
                                                                shuffle=True)

    labels = {value: key for key, value in train_generator.class_indices.items()}

    print("Label Mappings for classes present in the training and validation datasets\n")
    
    for key, value in labels.items():
        print(f"{key} : {value}")

    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(15, 12))
    idx = 0

    for i in range(2):
        for j in range(5):
            label = labels[np.argmax(train_generator[0][1][idx])]
            ax[i, j].set_title(f"{label}")
            ax[i, j].imshow(train_generator[0][0][idx][:, :, :])
            ax[i, j].axis("off")
            idx += 1

    plt.tight_layout()
    plt.suptitle("Sample Training Images", fontsize=20)
    plt.show()

    def create_model():
        model = Sequential([
            Conv2D(filters=64, kernel_size=(5, 5), padding='valid', input_shape=(IMG_WIDTH, IMG_WIDTH, 3)),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2), padding='valid'),
            BatchNormalization(),
            
            Conv2D(filters=128, kernel_size=(3, 3), padding='valid', kernel_regularizer=l2(0.00005)),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            BatchNormalization(),
        
            Conv2D(filters=128, kernel_size=(3, 3), padding='valid', kernel_regularizer=l2(0.00005)),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            BatchNormalization(),

            Conv2D(filters=256, kernel_size=(3, 3), padding='valid', kernel_regularizer=l2(0.00005)),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            BatchNormalization(),

            Flatten(),
            Dense(units=128, activation='sigmoid'),
            Dense(units=20, activation='softmax')
        ])
        
        return model

    cnn_model = create_model()
    print(cnn_model.summary())

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), patience=10)
    optimizer = Adam(learning_rate=0.001)
    cnn_model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(), metrics=['accuracy'])
    history = cnn_model.fit(train_generator, epochs=50, validation_data=validation_generator,
                        verbose=1,
                        callbacks=[reduce_lr])
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

    cnn_model.save('./trained_model.h5')