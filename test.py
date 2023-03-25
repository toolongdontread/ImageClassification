import argparse
import os
import sys
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import cv2
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_WIDTH = 112
IMG_HEIGHT = 112
BATCH_SIZE = 16

def create_arg_parser(): 
    parser = argparse.ArgumentParser()
    # test data folder
    parser.add_argument('data_path', 
                        type=Path, 
                        help='Test Data Path')
    # model
    parser.add_argument('model_path', 
                        type=Path, 
                        help='Trained Model Path')
    # output .txt file
    parser.add_argument('output_path', 
                        type=Path, 
                        help='Output Path')                                         
    return parser

# run the file in the cmd
if __name__ == "__main__":                                                        
    parser = create_arg_parser()
    args = parser.parse_args(sys.argv[1:])
    
    test_dataset = args.data_path
    cnn_model = tf.keras.models.load_model(args.model_path)

    # test
    test_datagen = ImageDataGenerator(horizontal_flip=True,
                                       zca_whitening=True,
                                       brightness_range=(0.8,1),
                                       channel_shift_range=60)
    test_generator = test_datagen.flow_from_directory(test_dataset,
                                                    batch_size=BATCH_SIZE,
                                                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                    color_mode='rgb')

    predictions = cnn_model.predict(test_generator)
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(12, 10))
    idx = 0
    labels = {value: key for key, value in test_generator.class_indices.items()}

    for key, value in labels.items():
            print(f"{key} : {value}")
    for i in range(2):
        for j in range(5):
            predicted_label = labels[np.argmax(predictions[idx])]
            ax[i, j].set_title(f"{predicted_label}")
            ax[i, j].imshow(test_generator[0][0][idx])
            ax[i, j].axis("off")
            idx += 1

    plt.tight_layout()
    plt.suptitle("Test Dataset Predictions", fontsize=20)
    plt.show()

    test_loss, test_accuracy = cnn_model.evaluate(test_generator, batch_size=BATCH_SIZE)
    print(f'Test Loss:     {test_loss}')
    print(f'Test Accuracy: {test_accuracy}')
    f = open(args.output_path, 'w')
    f.write(f'Test Loss:     {test_loss} \n')
    f.write(f'Test Accuracy: {test_accuracy}')
    f.close()

    # prediction
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    # print out the report
    print(classification_report(y_true, y_pred, target_names=labels.values()))
    
    '''
    # print out some wrong predictions
    errors = (y_true - y_pred != 0)
    y_true_errors = y_true[errors]
    y_pred_errors = y_pred[errors]
    test_images = test_generator.filenames
    test_img = np.asarray(test_images)[errors]

    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(12, 10))
    idx = 0

    for i in range(2):
        for j in range(5):
            idx = np.random.randint(0, len(test_img))
            true_index = y_true_errors[idx]
            true_label = labels[true_index]
            predicted_index = y_pred_errors[idx]
            predicted_label = labels[predicted_index]
            ax[i, j].set_title(f"True Label: {true_label} \n Predicted Label: {predicted_label}")
            img_path = os.path.join(test_dataset, test_img[idx])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax[i, j].imshow(img)
            ax[i, j].axis("off")

    plt.tight_layout()
    plt.suptitle('Wrong Predictions made on test set', fontsize=20)
    plt.show()
    '''