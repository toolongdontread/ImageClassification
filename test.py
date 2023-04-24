import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

    IMG_SIZE = 224
    BATCH_SIZE = 8

    parser = create_arg_parser()
    args = parser.parse_args(sys.argv[1:])
    
    test_dataset = args.data_path
    
    # test
    test_datagen = ImageDataGenerator(horizontal_flip=True,
                                       zca_whitening=True,
                                       zca_epsilon=1e-07,
                                       rescale=1/255,
                                       zoom_range=[0.7,1.1])

    cnn_model = tf.keras.models.load_model(args.model_path)    
    test_generator = test_datagen.flow_from_directory(test_dataset,
                                                    target_size=(IMG_SIZE, IMG_SIZE),
                                                    batch_size=BATCH_SIZE,
                                                    color_mode="rgb")
    predictions = cnn_model.predict(test_generator)
    
    test_loss, test_accuracy = cnn_model.evaluate(test_generator, batch_size=BATCH_SIZE)
    print(f"Test Loss:     {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")
    
    # prediction
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    f = open(args.output_path, 'w')
    f.write(f'Test Loss:        {test_loss}\n')
    f.write(f'Test Accuracy:    {test_accuracy}')
    f.close()
